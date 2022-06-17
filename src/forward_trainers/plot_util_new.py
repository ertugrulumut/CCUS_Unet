import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib.backends.backend_pdf import PdfPages


def plot_loss(df_loss, fontsize=28):

    fig_loss = plt.figure(figsize=(20, 10))
    plt.semilogy(df_loss['train_loss'], linestyle='dashed', color='red', marker='o')
    plt.semilogy(df_loss['val_loss'], linestyle='dashed', color='black', marker='o')
    plt.xlim(xmin=0)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    plt.xlim(xmin=0)
    plt.xlabel(f'Epoch', fontsize=fontsize)
    plt.ylabel(f'Loss by RMSE', fontsize=fontsize)
    plt.legend(['Train', 'Validation'], fontsize=fontsize)
    plt.title(f'Neural Net Performance', fontsize=fontsize)

    return fig_loss


def plot_parity(true_val, pred_val, tag, fontsize=28):

    r2 = r2_score(true_val, pred_val)

    # Randomly choose maximumally 10000 points
    num_points = len(true_val)
    points = np.arange(0, num_points)
    np.random.shuffle(points)
    selected_points = points[:10000]

    fig_parity = plt.figure(figsize=(20, 10))
    plt.plot(true_val[selected_points], pred_val[selected_points], 'ro')
    plt.plot(true_val[selected_points], true_val[selected_points], 'k-')
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    plt.xlabel(f'Simulation {tag}', fontsize=fontsize)
    plt.ylabel(f'DNN {tag}', fontsize=fontsize)
    plt.title(f'Neural Net Performance, r2 score of {tag}: {r2: .4f}', fontsize=fontsize)

    return fig_parity


def plot_time_loss(df_time_error, fontsize=28):

    fig_t_loss = plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(df_time_error['days'][1:], df_time_error['pres_loss'][1:], linestyle='solid', color='red', marker='o')
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel(f'Days', fontsize=fontsize)
    plt.ylabel(f'Pressure Loss', fontsize=fontsize)

    plt.subplot(1, 2, 2)
    plt.plot(df_time_error['days'][1:], df_time_error['sG_loss'][1:], linestyle='solid', color='black', marker='o')
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel(f'Days', fontsize=fontsize)
    plt.ylabel(f'Saturation Loss', fontsize=fontsize)

    return fig_t_loss


def generate_report(case_nums, data_true, data_pred, report_path, df_time_loss,
                    df_loss=None, fontsize=28):

    # Plot the results
    n_time_steps = data_true['n_time_steps']
    nx, ny, nz = data_true['n_grid_x'], data_true['n_grid_y'], data_true['n_grid_z']
    time_steps = data_true['timeSteps']
    selected_time_idx = [*range(n_time_steps)]
    selected_time_steps = [time_steps[j] for j in selected_time_idx]
    selected_layers = [0]

    with PdfPages(report_path) as pdf:

        # Plot parity chart
        fig_parity = plot_parity(
            true_val=data_true['pres_3d'][:, 1:, ...].flatten(),
            pred_val=data_pred['pres_3d'][:, 1:, ...].flatten(),
            tag='p (KPa)')
        pdf.savefig(fig_parity)
        plt.close()

        fig_parity= plot_parity(
            true_val=data_true['sG_3d'][:, 1:, ...].flatten(),
            pred_val=data_pred['sG_3d'][:, 1:, ...].flatten(),
            tag='Sg')
        pdf.savefig(fig_parity)
        plt.close()

        # Plot Neural Network Performance Results
        if df_loss is not None:
            fig_loss = plot_loss(df_loss, fontsize)
            pdf.savefig(fig_loss)
            plt.close()

        # Plot temporal error if available
        fig_t_loss = plot_time_loss(df_time_loss, fontsize)
        pdf.savefig(fig_t_loss)
        plt.close()

        # Loop each case
        for i, icase in enumerate(case_nums):

            if i > 0: continue  # For testing purpose

            # Get the permeability and porosity of each layer
            perm_3d = data_true.get('perm_3d', None)[i, :, :, :]
            perm_min, perm_max = np.nanmin(perm_3d), np.nanmax(perm_3d)

            # Plot each layers for pressure
            loc = 1
            fig_kphi = plt.figure(figsize=(30, 20))
            plt.suptitle(f'Permeability ($K$) Field of Case {icase+1}', fontsize=fontsize)
            for klayer in range(ny):

                # Skip not selected layers
                if klayer not in selected_layers:
                    continue

                loc += 1

                plt.subplot(3, 3, loc)
                plt.title(f'$K$ at layer {klayer + 1}', fontsize=fontsize)
                plt.imshow(perm_3d[:, klayer, :].T, vmin=perm_min, vmax=perm_max, cmap='jet', aspect='equal')
                cbar = plt.colorbar()
                cbar.set_label(f'$K, md$', labelpad=+1, y=1.07, rotation=0, fontsize=fontsize - 8)
                cbar.ax.tick_params(labelsize=fontsize - 8)

                loc += 1

            pdf.savefig(fig_kphi)
            plt.close()

            for j, jtstep in enumerate(time_steps):

                # Skip not selected timesteps
                if jtstep not in selected_time_steps:
                    continue

                print(f'Plot case {icase+1} at day = {jtstep}.')

                # Get ground truth
                pres_3d_gt = data_true['pres_3d'][i, j, :, :, :]
                sat_3d_gt = data_true['sG_3d'][i, j, :, :, :]
                pres_3d_pred = data_pred['pres_3d'][i, j, :, :, :]
                sat_3d_pred = data_pred['sG_3d'][i, j, :, :, :]

                # Get the difference
                pres_diff_3D = pres_3d_gt-pres_3d_pred
                sat_diff_3D = sat_3d_gt-sat_3d_pred

                # Get ranges
                pres_min = np.nanmin([np.nanmin(pres_3d_gt), np.nanmin(pres_3d_pred)])
                pres_max = np.nanmax([np.nanmax(pres_3d_gt), np.nanmax(pres_3d_pred)])
                sat_min = np.nanmin([np.nanmin(sat_3d_gt), np.nanmin(sat_3d_pred)])
                sat_max = np.nanmax([np.nanmax(sat_3d_gt), np.nanmax(sat_3d_pred)])

                # Get mean absolute error and relative error of pressure and water saturation at each timestep
                pres_abs_error_mean = np.nanmean(np.abs(pres_diff_3D))
                pres_rel_error_mean = np.nanmean(np.abs(pres_diff_3D) / (pres_max - pres_min))
                sat_abs_error_mean = np.nanmean(np.abs(sat_diff_3D))
                if sat_max == sat_min:
                    sG_rel_error_mean = 0.0
                else:
                    sG_rel_error_mean = np.nanmean(np.abs(sat_diff_3D) / (sat_max - sat_min))

                # Plot each layers for pressure
                loc = 0
                fig_pres = plt.figure(figsize=(30, 20))
                plt.suptitle(f'Pressure Map Case {icase+1} at {int(jtstep)} days,'
                             f'Mean Absolute Error = {pres_abs_error_mean: .3e} KPa,'
                             f'Mean Relative Error = {pres_rel_error_mean*100: .3e}%', fontsize=fontsize)

                for klayer in range(ny):

                    # Skip not selected layers
                    if klayer not in selected_layers:
                        continue

                    loc += 1
                    plt.subplot(3, 3, loc)
                    plt.title(f'Groud Truth at layer {klayer+1}', fontsize=fontsize)
                    plt.imshow(pres_3d_gt[:, klayer, :].T, vmin=pres_min, vmax=pres_max, cmap='jet', aspect='auto')
                    cbar = plt.colorbar()
                    cbar.set_label(f'$P, KPa$', labelpad=+1, y=1.07, rotation=0, fontsize=fontsize - 8)
                    cbar.ax.tick_params(labelsize=fontsize - 8)

                    loc += 1
                    plt.subplot(3, 3, loc)
                    plt.title(f'Prediction at layer {klayer+1}', fontsize=fontsize)
                    plt.imshow(pres_3d_pred[:, klayer, :].T, vmin=pres_min, vmax=pres_max, cmap='jet', aspect='auto')
                    cbar = plt.colorbar()
                    cbar.set_label(f'$P, KPa$', labelpad=+1, y=1.07, rotation=0, fontsize=fontsize - 8)
                    cbar.ax.tick_params(labelsize=fontsize - 8)

                    loc += 1
                    plt.subplot(3, 3, loc)
                    plt.title(f'Abs Difference at layer {klayer+1}', fontsize=fontsize)
                    plt.imshow(np.abs(pres_diff_3D[:, klayer, :].T), vmin=0, vmax=np.max(np.abs(pres_diff_3D)), cmap='jet', aspect='auto')
                    cbar = plt.colorbar()
                    cbar.set_label(f'$P, KPa$', labelpad=+1, y=1.07, rotation=0, fontsize=fontsize - 8)
                    cbar.ax.tick_params(labelsize=fontsize - 8)

                pdf.savefig(fig_pres)
                plt.close()

                # Plot each layers for saturation
                loc = 0
                fig_sat = plt.figure(figsize=(30, 20))
                plt.suptitle(f'CO2 Saturation Map Case {icase+1} at {int(jtstep)} days,'
                             f'Mean Absolute Error = {sat_abs_error_mean: .3e},'
                             f'Mean Relative Error = {sG_rel_error_mean*100: .3e}%', fontsize=fontsize)

                for klayer in range(ny):

                    # Skip not selected layers
                    if klayer not in selected_layers:
                        continue

                    loc += 1
                    plt.subplot(3, 3, loc)
                    plt.title(f'Groud Truth at layer {klayer+1}', fontsize=fontsize)
                    plt.imshow(sat_3d_gt[:, klayer, :].T, vmin=sat_min, vmax=sat_max, cmap='jet', aspect='auto')
                    cbar = plt.colorbar()
                    cbar.set_label(f'$S_g$', labelpad=+1, y=1.07, rotation=0, fontsize=fontsize - 8)
                    cbar.ax.tick_params(labelsize=fontsize - 8)

                    loc += 1
                    plt.subplot(3, 3, loc)
                    plt.title(f'Prediction of layer {klayer+1}', fontsize=fontsize)
                    plt.imshow(sat_3d_pred[:, klayer, :].T, vmin=sat_min, vmax=sat_max, cmap='jet', aspect='auto')
                    cbar = plt.colorbar()
                    cbar.set_label(f'$S_g$', labelpad=+1, y=1.07, rotation=0, fontsize=fontsize - 8)
                    cbar.ax.tick_params(labelsize=fontsize - 8)

                    loc += 1
                    plt.subplot(3, 3, loc)
                    plt.title(f'Abs Difference of layer {klayer+1}', fontsize=fontsize)
                    plt.imshow(np.abs(sat_diff_3D[:, klayer, :]), vmin=0.0, vmax=np.max(np.abs(sat_diff_3D)), cmap='jet', aspect='auto')
                    cbar = plt.colorbar()
                    cbar.set_label(f'$S_o$', labelpad=+1, y=1.07, rotation=0, fontsize=fontsize - 8)
                    cbar.ax.tick_params(labelsize=fontsize - 8)

                pdf.savefig(fig_sat)
                plt.close()


