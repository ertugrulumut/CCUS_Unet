import copy
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from kernel.unet_2d import Unet2d
from forward_trainers.plot_util_new import generate_report
from wrangler.data_parser import load_from_h5, save_to_h5, load_from_h5_general
from forward_trainers.stats import create_scalers, save_scaler_meta
from torch_util_new import get_device, RMSELoss as criteria, count_parameters, \
    get_datetime, get_temporal_loss, get_loss, train_val_test_split

np.random.seed(0)
torch.manual_seed(0)

class CNN():

    def __init__(self,
                 learning_rate,
                 weight_decay,
                 num_epochs,
                 max_stagnancy,
                 in_channel,
                 dataloaders,
                 scaler_dict,
                 device,
                 file_paths,
                 iter_verbose=True,
                 epoch_verbose=True
                 ):

        # Save the data scaler
        self.scaler_dict = scaler_dict

        # Save model path
        self.file_paths = file_paths

        # Save dataloader
        self.dataloaders = dataloaders

        # Get device
        if device:
            self.device = device
        else:
            self.device = get_device('GPU')

        # Set maxmin stagnancy for Adam optimizer
        self.max_stagnancy = max_stagnancy

        self.num_epochs = num_epochs

        # Save loss changes
        self.ds_loss = []

        # deep neural networks last layer
        self.softplus = torch.nn.Softplus() # Ensure positiveness
        self.relu = torch.nn.ReLU()         # Ensure positiveness or apply upstream weighting
        self.silu = torch.nn.SiLU()
        self.sigmoid = torch.nn.Sigmoid()

        # Set up pressure network and saturation network
        self.pnet = Unet2d(in_channels=8, out_channels=1, init_features=32).to(self.device)
        self.snet = Unet2d(in_channels=8, out_channels=1, init_features=32).to(self.device)
        self.num_model_params = count_parameters(self.pnet) + count_parameters(self.snet)

        # Optimizers: optimizing pnet and snet
        self.pnet_optimizer = torch.optim.Adam(
            self.pnet.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.snet_optimizer = torch.optim.Adam(
            self.snet.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.pnet_scheduler = torch.optim.lr_scheduler.StepLR(self.pnet_optimizer, step_size=100, gamma=0.5)
        self.snet_scheduler = torch.optim.lr_scheduler.StepLR(self.snet_optimizer, step_size=100, gamma=0.5)

        # Verbose control
        self.iter_verbose = iter_verbose
        self.epoch_verbose = epoch_verbose


    def create_dataset(self, X, Y):
        """Prepare input and output data for NNs."""

        # Get all the input data
        self.X_next = X[:, :, :, :].to(self.device)
        self.p_next_true_scaled = Y[:, 0, :, :].to(self.device)
        self.Sw_next_true_scaled = Y[:, 1, :, :].to(self.device)


    def predict_P(self, X_pnet):
        Y = self.pnet(X_pnet)[..., 0]
        Y = self.silu(Y)
        return Y


    def predict_Sw(self, X_snet):
        Y = self.snet(X_snet)[..., 0]
        Y = self.silu(Y)
        return Y


    def train(self, always_train=True):
        """Train both pressure and saturation network."""

        if self.file_paths['pnet'].exists():
            self.pnet.load_state_dict(torch.load(self.file_paths['pnet']))
            if always_train:
                self.pnet_trainer()
        else:
            self.pnet_trainer()

        if self.file_paths['snet'].exists():
            self.snet.load_state_dict(torch.load(self.file_paths['snet']))
            if always_train:
                self.snet_trainer()
        else:
            self.snet_trainer()

    def standalone_train(self):
        """Train pnet and snet."""
        self.pnet_trainer()
        self.snet_trainer()


    def pnet_trainer(self):

        # Train with Adam optimizer
        stagnancy_pnet = 0
        best_loss_p = 1.0e10
        best_pnet_wts = copy.deepcopy(self.pnet.state_dict())

        # df column names
        df_column = ['epoch']
        for phase in ['train', 'val']: #
            for col in ['loss_p']:
                df_column.append(f'{phase}_{col}')
        df_column.append('cpu')

        # Epoch iteration
        for epoch in range(1, self.num_epochs + 1):

            # Epoch recorder
            epoch_cpu = time.time()
            epoch_record = [epoch]

            for phase in ['train' , 'val']:

                total_iterations = len(self.dataloaders[phase])
                epoch_samples = 0
                epoch_loss_p = 0.0

                if phase == 'train':
                    self.pnet.train()
                else:
                    self.pnet.eval()

                for iteration, (X, Y) in enumerate(self.dataloaders[phase], 1):

                    # Setup timer
                    batch_cpu = time.time()
                    batch_size = X.shape[0]

                    # Create dataset
                    self.create_dataset(X, Y)

                    # Zero the parameter gradients of each optimizer
                    self.pnet_optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase in ['train']):

                        # Perform prediction
                        p_next_pred_scaled = self.predict_P(self.X_next)
                        loss_p = criteria(p_next_pred_scaled, self.p_next_true_scaled)

                        if phase == 'train':
                            loss_p.backward(retain_graph=True)
                            self.pnet_optimizer.step()

                    # End timer
                    batch_cpu = time.time() - batch_cpu

                    # Iteration verbose
                    if self.iter_verbose:
                        print(f'Epoch {phase.upper()}: {epoch}/{self.num_epochs}, Iter: {iteration}/{total_iterations}, batch size: {batch_size}, '
                              f'loss: pres: {loss_p.item(): .2e}, '
                              f'CPU Time: {batch_cpu: .3f} sec, environ: {str(self.device).upper()}.')

                    # Accumulate epoch recorder
                    epoch_samples += batch_size
                    epoch_loss_p += loss_p.cpu().item() * batch_size

                # Update learning rate
                if phase == 'train':
                    self.pnet_scheduler.step()
                    for param_group in self.pnet_optimizer.param_groups:
                        pnet_lr = param_group["lr"]

                # Average out epoch record
                epoch_loss_p = epoch_loss_p / epoch_samples

                # Update epoch_record
                epoch_record.extend([epoch_loss_p])

                # Check stagnancy during validation
                if phase == 'val':
                    if (epoch_loss_p < best_loss_p):
                        best_loss_p = epoch_loss_p
                        best_pnet_wts = copy.deepcopy(self.pnet.state_dict())
                        self.save(models='pnet')
                        stagnancy_pnet = 0
                    else:
                        stagnancy_pnet += 1

            # Save the loss
            epoch_cpu = time.time() - epoch_cpu
            epoch_record.append(epoch_cpu)
            self.ds_loss.append(epoch_record)
            self.save_loss(df_column, key='ploss')

            # Epoch verbose
            if self.epoch_verbose:
                print(f'---\n')
                print(f'Epoch: {epoch}/{self.num_epochs}, '
                      f'{total_iterations} batches, {int(epoch_samples/total_iterations)} samples/batch, '
                      f'loss pres: {epoch_loss_p: .2e}.\n'
                      f'pnet lr: {pnet_lr}, stagnancy: {stagnancy_pnet}/{self.max_stagnancy}, '
                      f'best p loss: {best_loss_p: .2e}, '
                      f'CPU Time: {epoch_cpu: .3f} sec, environ: {str(self.device).upper()}.')
                print(f'---\n')

            if self.iter_verbose:
                print('*' * 200)

            # Exit if max stagnancy is met
            if (stagnancy_pnet >= self.max_stagnancy):
                print(f'pnet stagnancy = {stagnancy_pnet}/{self.max_stagnancy}, stop train pnet for reaching max stagnancy.')
                break

        # Load best model weights
        print(f'Best p loss: {best_loss_p: .2e}')
        self.pnet.load_state_dict(best_pnet_wts)


    def snet_trainer(self):

        # Train with Adam optimizer
        stagnancy_snet = 0
        best_loss_Sw = 1.0e10
        best_snet_wts = copy.deepcopy(self.snet.state_dict())

        # df column names
        df_column = ['epoch']
        for phase in ['train' , 'val']: #
            for col in ['loss_Sw']:
                df_column.append(f'{phase}_{col}')
        df_column.append('cpu')

        # Epoch iteration
        for epoch in range(1, self.num_epochs + 1):

            # Epoch recorder
            epoch_cpu = time.time()
            epoch_record = [epoch]

            for phase in ['train', 'val']:

                total_iterations = len(self.dataloaders[phase])
                epoch_samples = 0
                epoch_loss_Sw = 0.0

                if phase == 'train':
                    self.snet.train()
                else:
                    self.snet.eval()

                for iteration, (X, Y) in enumerate(self.dataloaders[phase], 1):

                    # Setup timer
                    batch_cpu = time.time()
                    batch_size = X.shape[0]

                    # Create dataset
                    self.create_dataset(X, Y)

                    # Zero the parameter gradients of each optimizer
                    self.snet_optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase in ['train']):

                        # Perform prediction
                        Sw_next_pred_scaled = self.predict_Sw(self.X_next)
                        loss_Sw = criteria(Sw_next_pred_scaled, self.Sw_next_true_scaled)

                        if phase == 'train':
                            loss_Sw.backward(retain_graph=True)
                            self.snet_optimizer.step()

                    # End timer
                    batch_cpu = time.time() - batch_cpu

                    # Iteration verbose
                    if self.iter_verbose:
                        print(f'Epoch {phase.upper()}: {epoch}/{self.num_epochs}, '
                              f'Iter: {iteration}/{total_iterations}, batch size: {batch_size}, '
                              f'loss Sw:{loss_Sw.item(): .2e}, '
                              f'CPU Time: {batch_cpu: .3f} sec, environ: {str(self.device).upper()}.')

                    # Accumulate epoch recorder
                    epoch_samples += batch_size
                    epoch_loss_Sw += loss_Sw.cpu().item() * batch_size

                # Update learning rate
                if phase == 'train':
                    self.snet_scheduler.step()
                    for param_group in self.snet_optimizer.param_groups:
                        snet_lr = param_group["lr"]

                # Average out epoch record
                epoch_loss_Sw = epoch_loss_Sw / epoch_samples

                # Update epoch_record
                epoch_record.extend([epoch_loss_Sw])

                # Check stagnancy during validation
                if phase == 'val':
                    if (epoch_loss_Sw < best_loss_Sw):
                        best_loss_Sw = epoch_loss_Sw
                        best_snet_wts = copy.deepcopy(self.snet.state_dict())
                        self.save(models='snet')
                        stagnancy_snet = 0
                    else:
                        stagnancy_snet += 1

            # Save the loss
            epoch_cpu = time.time() - epoch_cpu
            epoch_record.append(epoch_cpu)
            self.ds_loss.append(epoch_record)
            self.save_loss(df_column, key='sloss')

            # Epoch verbose
            if self.epoch_verbose:
                print(f'---\n')
                print(f'Epoch: {epoch}/{self.num_epochs}, '
                      f'{total_iterations} batches, {int(epoch_samples/total_iterations)} samples/batch, '
                      f'loss: Sw: {epoch_loss_Sw: .2e}.\n'
                      f'snet lr: {snet_lr}, stagnancy: {stagnancy_snet}/{self.max_stagnancy}, '
                      f'best Sw loss: {best_loss_Sw: .2e}, '
                      f'CPU Time: {epoch_cpu: .3f} sec, environ: {str(self.device).upper()}.')
                print(f'---\n')

            if self.iter_verbose:
                print('*' * 200)

            # Exit if max stagnancy is met
            if (stagnancy_snet >= self.max_stagnancy):
                print(f'snet stagnancy = {stagnancy_snet}/{self.max_stagnancy}, stop train snet for reaching max stagnancy.')
                break

        # Load best model weights
        print(f'Best Sw loss: {best_loss_Sw: .2e}.')
        self.snet.load_state_dict(best_snet_wts)


    def save(self, models='all'):
        if models == 'all':
            torch.save(self.pnet.state_dict(), self.file_paths['pnet'])
            torch.save(self.snet.state_dict(), self.file_paths['snet'])
        if models == 'pnet':
            torch.save(self.pnet.state_dict(), self.file_paths['pnet'])
        if models == 'snet':
            torch.save(self.snet.state_dict(), self.file_paths['snet'])
        return


    def save_loss(self, df_column, key='loss'):
        """Key can loss, ploss, sloss."""
        df_loss = pd.DataFrame(data=self.ds_loss, columns=df_column)
        df_loss.to_csv(self.file_paths[key], index=False)
        return

def get_secondary_variables(data):
    """Get some well related variables."""

    new_data = data.copy()
    # Get inactive grid filter
    inactive_grid_filter = np.zeros_like(new_data['perm_3d'])
    inactive_grid_filter[new_data['perm_3d'] > 0.0] = 1.0
    new_data['inactive_grid_filter'] = inactive_grid_filter

    return new_data

def data_transform(data, cases, torchTensor=True):
    """Transform the data for Deep Learning."""

    n_grid_x = data.get('n_grid_x')
    n_grid_y = data.get('n_grid_y')
    n_grid_z = data.get('n_grid_z')
    n_samples = data.get('n_samples')
    n_time_steps = data.get('n_time_steps')

    # Sample loop
    features, labels = [], []

    # Input and output channels for both saturation and pressure network
    in_channels = 8
    out_channels = 2

    # Sample loop
    for isample in range(n_samples):

        # Time step loop: offset by 1
        for next_tstep in range(1, n_time_steps, 1):

            current_tstep = next_tstep - 1

            # Layer loop
            for ilayer in range(n_grid_y):

                # Init memory
                feature = np.zeros(shape=(n_grid_x, n_grid_z, in_channels), dtype=np.float32)
                label = np.zeros(shape=(out_channels, n_grid_x, n_grid_z), dtype=np.float32)

                # Get the features of pressure and saturation
                feature[:, :, 0] = data.get('perm_3d')[isample, :, ilayer, :]
                feature[:, :, 1] = data.get('pore_vol')[isample, :, ilayer, :]
                feature[:, :, 2] = data.get('Reservoir_thickness')[isample]
                feature[:, :, 3] = data.get('Temperature')[isample]
                feature[:, :, 4] = data.get('Caprock_thickness')[isample]
                feature[:, :, 5] = data.get('Caprock_Cohesion')[isample]
                feature[:, :, 6] = data.get('InjRate_Perf')[isample, :, ilayer, :]
                feature[:, :, 7] = data.get('timeStep_ts')[next_tstep]

                # Get the labels of pressure and saturation
                label[0, :, :] = data.get('pres_3d')[isample, next_tstep, :, ilayer, :]
                label[1, :, :] = data.get('sG_3d')[isample, next_tstep, :, ilayer, :]

                # Merge into test, validation and train cases
                if isample in cases:
                    labels.append(label)
                    features.append(feature)

    # Convert to numpy array
    labels = np.array(labels, dtype=np.float32)
    features = np.array(features, dtype=np.float32)

    if torchTensor:
        labels = torch.from_numpy(labels)
        features = torch.from_numpy(features)

    return features, labels, in_channels, out_channels

def sequential_prediction(data, cases, model, scaler_dict, device):
    """Transform the data for Deep Learning."""

    n_samples = len(cases)
    n_grid_x = data.get('n_grid_x')
    n_grid_y = data.get('n_grid_y')
    n_grid_z = data.get('n_grid_z')
    n_time_steps = data.get('n_time_steps')

    # Activate models for prediciton
    model.pnet.eval()
    model.snet.eval()

    # Input and output channels for saturation and pressure networks
    in_channels = 8

    # Initialize p and S at prediction
    p_pred = np.zeros(shape=(n_samples, n_time_steps, n_grid_x, n_grid_y, n_grid_z), dtype=float)
    s_pred = np.zeros_like(p_pred)

    # Init memory
    feature = np.zeros(shape=(1, n_grid_x, n_grid_z, in_channels), dtype=np.float32)

    # Sample loop
    for i, isample in enumerate(cases):

        # p or S at the time step 0
        p_pred[i, 0, :, 0, :] = data.get('pres_3d')[isample, 0, :, 0, :]
        s_pred[i, 0, :, 0, :] = data.get('sG_3d')[isample, 0, :, 0, :]

        # Update features
        feature[0, :, :, 0] = data.get('perm_3d')[isample, :, 0, :]
        feature[0, :, :, 1] = data.get('pore_vol')[isample, :, 0, :]
        feature[0, :, :, 2] = data.get('Reservoir_thickness')[isample]
        feature[0, :, :, 3] = data.get('Temperature')[isample]
        feature[0, :, :, 4] = data.get('Caprock_thickness')[isample]
        feature[0, :, :, 5] = data.get('Caprock_Cohesion')[isample]
        feature[0, :, :, 6] = data.get('InjRate_Perf')[isample, :, 0, :]

        # Time step loop: offset by 1
        for next_tstep in range(1, n_time_steps, 1):

            print(f'Predict time step {(next_tstep+1)}/{n_time_steps} of '
                  f'sample {(isample+1)}/{len(cases)} (case {isample})')

            # Dynamic update p and S at current step based on prediction
            # except the first time step
            feature[0, :, :, 7] = data.get('timeStep_ts')[next_tstep]

            # Perform prediction
            p_pred[i, next_tstep, :, 0, :] = model.predict_P(
                torch.tensor(feature).to(device)).detach().cpu().numpy()[0, :, :]
            s_pred[i, next_tstep, :, 0, :] = model.predict_Sw(
                torch.tensor(feature).to(device)).detach().cpu().numpy()[0, :, :]

    # Inversely scale the results
    p_pred = scaler_dict['pres_3d'].inverse_transform(p_pred)
    s_pred = scaler_dict['sG_3d'].inverse_transform(s_pred)

    # Check bounds of pressure and saturation
    s_pred[s_pred < 0.0] = 0.0
    s_pred[s_pred > 1.0] = 1.0
    p_pred[p_pred < 0.0] = 0.0

    return {
        'pres_3d': p_pred,
        'sG_3d': s_pred
    }

def select_data_by_case_number(data, cases):
    """Select only cases in the data dictionary."""

    # nan inactive grids
    perm_3d = data['perm_3d'][cases, ...]
    pres_3d = data['pres_3d'][cases, ...]
    sG_3d = data['sG_3d'][cases, ...]

    perm_3d[perm_3d == 0.0] = np.nan
    pres_3d[pres_3d == 0.0] = np.nan
    sG_3d[sG_3d == 0.0] = np.nan

    return {
        'n_samples': len(cases),
        'n_time_steps': data['n_time_steps'],
        'n_grid_x': data['n_grid_x'],
        'n_grid_y': data['n_grid_y'],
        'n_grid_z': data['n_grid_z'],
        'perm_3d': perm_3d,
        'inactive_grid_filter': data['inactive_grid_filter'][cases, ...],
        'pres_3d': pres_3d,
        'sG_3d': sG_3d,
        'timeSteps': data['timeStep_ts'],
    }

def engine(payload, data):
    """Main function to perform training and prediction."""

    # Download the control parameters
    test_cases = payload['test_cases']
    train_cases = payload['train_cases']
    validation_cases = payload['validation_cases']

    device = get_device(payload.get('device', 'cpu'))
    batch_size = payload.get('batch_size', 10)
    num_epochs = payload.get('num_epochs', 100)
    max_stagnancy = payload.get('max_stagnancy', 100)
    lr = payload.get('lr', 0.01)
    wd = payload.get('wd', 1e-4)

    output_path = payload['output_path']
    model_file = payload['model_file']
    model_meta_file = payload['model_meta_file']
    scaler_meta_file = payload['scaler_meta_file']
    pdf_report = payload['pdf_report']
    temporal_loss_file = payload['temporal_loss_file']
    data_true_file = payload['data_true_file']
    data_pred_file = payload['data_pred_file']

    # Init paths
    Path(output_path).mkdir(exist_ok=True)
    pdf_report_path = Path(output_path, pdf_report)
    model_meta_path = Path(output_path, model_meta_file)
    scaler_meta_path = Path(output_path, scaler_meta_file)
    temporal_loss_path = Path(output_path, temporal_loss_file)
    data_true_path = Path(output_path, data_true_file)
    data_pred_path = Path(output_path, data_pred_file)

    # Save some meta data for the model
    model_meta = {
        'start_datetime': get_datetime(),
        'device': str(device),
        'test_cases': test_cases,
        'train_cases': train_cases,
        'validation_cases': validation_cases,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'max_stagnancy': max_stagnancy,
        'learning_rate': lr,
        'weight_decay': wd,
    }

    with open(model_meta_path, 'w') as config_file:
        json.dump(model_meta, config_file, indent=4)

    # Get well related data
    data = get_secondary_variables(data)

    # Scale the data first
    scaler_dict, data_scaled = create_scalers(data)

    # Save the scaler
    save_scaler_meta(scaler_dict, scaler_meta_path)

    # Train pressure and saturation model separately
    models = {}
    cpu_train = time.time()

    # Get file path
    file_paths = {
        'pnet': Path(output_path, f'pnet_{model_file}'),
        'snet': Path(output_path, f'snet_{model_file}'),
        'loss': Path(output_path, f'df_co_loss.csv'),
        'ploss': Path(output_path, f'df_p_loss.csv'),
        'sloss': Path(output_path, f'df_s_loss.csv'),
    }

    # Transform the data and wrap as loader
    x_train, y_train, in_channel, out_channels = data_transform(data_scaled, train_cases, torchTensor=True)
    x_val, y_val, _, _ = data_transform(data_scaled, validation_cases, torchTensor=True)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_val, y_val),
        batch_size=batch_size, shuffle=True)

    # Dataloader for train and validation data
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Setup CNN
    model = CNN(lr, wd, num_epochs, max_stagnancy, in_channel, dataloaders, scaler_dict, device, file_paths,
                iter_verbose=True, epoch_verbose=True)
    model.train(always_train=False)

    # Update models, model_test_dataloader and model_meta
    models.update({
        'pnet': model.pnet,
        'snet': model.snet,
    })

    # Get training time
    cpu_train = time.time() - cpu_train

    model_meta.update({
        f'num_model_params': model.num_model_params,
        f'cpu_train(sec)': cpu_train,
    })

    #############################################
    # Prediction for test dataset
    #############################################
    # Perform preditions
    cpu_pred = time.time()

    # Predict in a sequential mode
    data_pred = sequential_prediction(data_scaled, test_cases, model, scaler_dict, device)

    # Get only testing case data of ground truth
    data_true = select_data_by_case_number(data, test_cases)
    cpu_pred = time.time() - cpu_pred
    model_meta.update({f'cpu_pred(sec)': cpu_pred})

    # delete some memory
    del data_scaled, data

    #############################################
    # Plotting results and generate pdf report
    #############################################
    df_time_loss = get_temporal_loss(data_true, data_pred)
    df_time_loss.to_csv(temporal_loss_path, index=False)
    losses = get_loss(data_true, data_pred)

    # Generate pdf report
    generate_report(test_cases, data_true, data_pred, pdf_report_path,
                    df_time_loss, df_loss=None)

    model_meta.update(losses)
    with open(model_meta_path, 'w') as config_file:
        json.dump(model_meta, config_file, indent=4)

    # Save prediction and true results
    save_to_h5(data_true, data_true_path)
    save_to_h5(data_pred, data_pred_path)


if __name__ == '__main__':

    # Load the simulation data
    cpu_time = time.time()
    data = load_from_h5_general(Path(f'C:\\Users\\yildireu\\PycharmProjects\\CCUS_DL_Gym\\sim_input\CCUS_Radial_Parfor.h5'))
    cpu_time = time.time() - cpu_time
    print(f'CPU time to load the simulation data: {cpu_time: .2f} seconds.')

    # Grid sizes in x-direction
    delta_x = np.zeros((50, 1, 50))
    delta_x[0, 0, :] = data['grid_xcoord'][0, 0, :] * 2
    for i in range(len(data['grid_xcoord']) - 1):
        delta_x[i + 1, 0, :] = (data['grid_xcoord'][i + 1, 0, 0] - np.sum(delta_x[:i + 1, 0, 0])) * 2
    # Grid sizes in z-direction
    delta_z = np.zeros((50, 1, 50))
    delta_z[:, 0, 0] = data['grid_zcoord'][:, 0, 0] * 2
    for i in range(len(data['grid_zcoord']) - 1):
        delta_z[:, 0, i + 1] = (data['grid_zcoord'][0, 0, i + 1] - np.sum(delta_z[0, 0, :i + 1])) * 2
    # Volume of each grid cell
    volume = delta_x * delta_z
    # Pore Volume of each grid cell
    pore_vol = np.zeros_like(data['poro_3d'])
    for k in range(data['poro_3d'].shape[0]):
        pore_vol[k] = volume * data['poro_3d'][k]

    # Combine Injection Rate and Perforation Depth
    InjRate_Perf = np.zeros_like(data['poro_3d'])
    for i in range(len(InjRate_Perf)):
        perf = data['Perforations'][i].astype(int)
        inj = data['Injection_rate'][i]
        InjRate_Perf[i, 0, 0, perf:] = inj / (50 - perf)

    data.update({
        'n_samples': data['perm_3d'].shape[0],
        'n_grid_x': data['perm_3d'].shape[1],
        'n_grid_y': data['perm_3d'].shape[2],
        'n_grid_z': data['perm_3d'].shape[3],
        'n_time_steps': data['timeStep_ts'].shape[0],
        'pore_vol': pore_vol,
        'InjRate_Perf': InjRate_Perf
    })


    # First select train/validation/test cases
    train_cases, test_cases, validation_cases = train_val_test_split(
        nsamples=data['n_samples'],
        test_size=0.1, val_size=0.1)

    # Define Payload
    payload = {
        'net_option': 'PICNN',
        'device': 'gpu',                 # Computing environment setup: 'cpu' OR 'gpu'
        'test_cases': test_cases,
        'train_cases': train_cases,
        'validation_cases': validation_cases,
        'batch_size': 20,                # Batch size for dataloader: default: 25
        'num_epochs': 100,               # Max number of epochs
        'max_stagnancy': 10,             # Max stagnant epochs to stop training
        'model_file': 'cnn.pth',
        'loss_file': 'cnn_loss.csv',
        'temporal_loss_file': 'temporal_loss.csv',
        'pdf_report': 'report.pdf',
        'data_true_file': 'data_true.h5',
        'data_pred_file': 'data_pred.h5',
        'model_meta_file': 'model_meta.json',
        'scaler_meta_file': 'scaler_meta.json',
        'lr': 1e-4,
        'wd': 1e-4,
        'output_path': Path(f'C:\\Users\\yildireu\\PycharmProjects\\CCUS_Unet\\output_unet'),
    }

    # Call the engine
    engine(payload, data)
