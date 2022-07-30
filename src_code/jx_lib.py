import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

from typing import Dict, Any, List

def get_files(DIR:str, file_end:str=".png"):
    return [ os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith(file_end) ]

def create_all_folders(DIR:str):
    path_ = ""
    for folder_name_ in DIR.split("/"):
        path_ = os.path.join(path_, folder_name_)
        create_folder(path_, False)

def clean_folder(DIR:str):
    create_folder(DIR=DIR, clean=True)

def create_folder(DIR:str, clean:bool=False):
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    elif clean:
        filelist = get_files(DIR)
        for f in filelist:
            os.remove(f)

def make_confusion_matrix(
    cf,
    group_names=None,
    categories='auto',
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    figsize=None,
    cmap='Blues',
    title=None
):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    https://github.com/DTrimarchi10/confusion_matrix
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    status = {}
    status2 = {}
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        status["Accuracy" ]  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            status["Precision"] = cf[1,1] / sum(cf[:,1])
            status["Recall"   ] = cf[1,1] / sum(cf[1,:])
            status["F1 Score" ] = 2 * status["Precision"] * status["Recall"] / (status["Precision"] + status["Recall"])
            
            status2["SP"] = cf[1,1] / sum(cf[:,1])
            status2["PP"] = cf[1,1] / sum(cf[1,:])
            status2["SN"] = cf[0,0] / sum(cf[:,0])
            status2["PN"] = cf[0,0] / sum(cf[0,:])
            status2["Score"] = 6 * status2["SP"]+ 5 * status2["SN"]+ 3 * status2["PP"]+2 * status2["PN"]

    stats_text = "\n\n" + " | ".join(["{:10s}={:.3f}".format(key, status[key]) for key in status])
    stats2_text = "\n" + " | ".join(["{:6s}={:.3f}".format(key, status2[key]) for key in status2])

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)
    ax.set_aspect(1)
    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text + stats2_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)
    
    return fig, status


def make_comparison_matrix(
    dict_of_status_log,
    report_method="best",
    xlabel="",
    ylabel="",
    cbar=True,
    figsize=None,
    cmap='Blues',
    title=None
):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    '''

    y_header = list(dict_of_status_log)
    x_header = list(dict_of_status_log[y_header[0]])
    entry_list = list(dict_of_status_log[y_header[0]][x_header[0]])
    sqr = np.ceil(np.sqrt(len(entry_list)))

    fig = plt.figure(figsize=figsize)
    for i, entry in enumerate(entry_list):
        # fetch data
        data = np.zeros((len(y_header), len(x_header)))
        for j,x in enumerate(x_header):
            for k,y in enumerate(y_header):
                if report_method == 'best':
                    data[k, j] = np.max(dict_of_status_log[y][x][entry])
                elif report_method == 'worst':
                    data[k, j] = np.min(dict_of_status_log[y][x][entry])
                elif report_method == 'average':
                    data[k, j] = np.average(dict_of_status_log[y][x][entry])
                else:
                    raise ValueError("Only 'best/worst/average' is implemented!")
                    

        group_labels = ["{:.2f}".format(value) for value in data.flatten()]
        box_labels = np.asarray(group_labels).reshape(data.shape[0], data.shape[1])

        # MAKE THE HEATMAP VISUALIZATION
        ax = plt.subplot(sqr,sqr,i+1)
        sns.heatmap(data,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=x_header,yticklabels=y_header)
        ax.set_aspect(1)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title("{} ({})".format(entry, report_method))
        
    return fig

def pie_plot(
        labels,
        sizes,
        title,
        figsize=(6,6),
        startangle=90,
        shadow=False
    ):
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=shadow, startangle=startangle)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)
    return fig

def imgs_plot(
        dict_of_imgs,
        figsize = (6,6),
        cmap    = None,
        OUT_DIR = "",
        tag     = "",
        show    = False
    ):
    fig = plt.figure(figsize=figsize)
    sqr = np.ceil(np.sqrt(len(dict_of_imgs)))

    for i,label in enumerate(dict_of_imgs):
        ax = plt.subplot(sqr,sqr,i+1)
        ax.imshow(dict_of_imgs[label], cmap=cmap)
        plt.xlabel(label)

    plt.tight_layout()
    fig.savefig("{}/plot_{}.png".format(OUT_DIR, tag), bbox_inches = 'tight')
    if not show:
        plt.close(fig)
    return fig


def progress_plot(
    h,
    figsize=(6,6)
):
    xs = list(range(1, 1+len(h.history['accuracy'])))
    # Plot
    fig = plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    plt.plot(xs, h.history['accuracy'], label="training")
    plt.plot(xs, h.history['val_accuracy'], label="validation")
    plt.ylabel("Accuracy")
    plt.xlabel("epoch")
    plt.xticks(xs)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(xs, h.history['loss'], label="training")
    plt.plot(xs, h.history['val_loss'], label="validation")
    plt.xticks(xs)
    plt.ylabel("Loss (cross-entropy)")
    plt.xlabel("epoch")
    plt.legend()

    return fig

def output_plot(
    data_dict,
    Ylabel  = "",
    Xlabel  = "",
    figsize = (12,6),
    OUT_DIR = "",
    tag     = ""
):
    fig = plt.figure(figsize=figsize)
    for name_, data_ in data_dict.items():
        plt.plot(data_["x"], data_["y"], label=name_) 
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.legend()
    plt.title("Plot [{}]".format(tag))
    fig.savefig("{}/plot_{}.png".format(OUT_DIR, tag), bbox_inches = 'tight')
    plt.close(fig)
    return fig


def output_prediction_result_plot(
    labels,
    dict_input_x,
    dict_prob,
    figsize = (12,6),
    OUT_DIR = "",
    tag     = ""
):
    for test_name in dict_prob:
        fig = plt.figure(figsize=figsize)
        # pie : prediction percentage
        ax = plt.subplot(1, 2, 1)
        predict_label = np.argmax(dict_prob[test_name])
        explode = np.zeros(len(labels))
        explode[predict_label] = 0.1
        ax.pie(dict_prob[test_name], labels=tuple(labels), autopct='%1.1f%%', explode=explode)
        plt.xlabel("Prediction Confidence (Probability)")
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # test img
        ax = plt.subplot(1, 2, 2)
        ax.imshow(dict_input_x[test_name])
        plt.xlabel("Input Test Image Data [{}]".format(test_name))
        fig.savefig("{}/test_sample_prediction_{}[{}].png".format(OUT_DIR, tag, test_name), bbox_inches = 'tight')
        plt.close(fig)

    return fig

def output_hist(
    data_dict,
    figsize = (12,6),
    bin_size= 20,
    OUT_DIR = "",
    tag     = ""
):
    fig = plt.figure(figsize=figsize)
    for name_, data_ in data_dict.items():
        plt.hist(data_, density=True, bins=bin_size, label=name_, alpha=1/len(data_dict)) 
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.legend()
    plt.title("Data Distribution [{}]".format(tag))
    fig.savefig("{}/hist_{}.png".format(OUT_DIR, tag), bbox_inches = 'tight')
    plt.close(fig)
    return fig

def dict_params_to_str_format(params:Dict[str, float]):
    return "[{}]".format(", ".join(["%s: %.3f"%(y,x) if np.shape(x) == () else "%s: [...]"%(y) for y, x in params.items()]))

def output_prediction(
    data_dict                : Dict[str, Any],
    model_truth_dict         : Dict[str, Any],
    dataset_label_dict       : Dict[str, Any],
    model_param_truth_dict   : Dict[str, Any],
    model_predict_dict       : Dict[str, Any],
    model_param_predict_dict : Dict[str, Any],
    hist                     : bool            = False,
    figsize                  : tuple           = (12,6),
    bin_size                 : int             = 20,
    OUT_DIR                  : str             = "",
    tag                      : str             = "",
    xlim                     : List[int]       = [0, 10],
    verbose                  : bool            = False,
) -> "figure":
    left,right = xlim
    fig = plt.figure(figsize=figsize)
    for name_, data_ in data_dict.items():
        model_true_func = model_truth_dict[name_]
        model_estimate_func = model_predict_dict[name_]
        data_label = dataset_label_dict[name_]
        # plot histo.
        if hist:
            plt.hist(data_, density=True, bins=bin_size, label="{}-histogram".format(data_label))
            left, right = plt.xlim()
        x = np.arange(left, right, step=0.01)
        # plot true
        param = model_param_truth_dict[name_]
        y = model_true_func(x=x, params=param)
        # plot prediction
        plt.plot(x, y, '--', label="p-true: {}".format(dict_params_to_str_format(param)))
        param = model_param_predict_dict[name_]
        y = model_estimate_func(x=x, params=param)
        plt.plot(x, y, label="p-estimate: {}".format(dict_params_to_str_format(param)))
        
    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.legend()
    plt.title("Data Distribution [{}]".format(tag))
    fig.savefig("{}/predict_{}.png".format(OUT_DIR, tag), bbox_inches = 'tight')
    if verbose:
        plt.show()
    plt.close(fig)
    return fig

def plot_probabilistic_contour(params, color, label, plot_axis=False, n_contours=1):
    mean, S = params["mean"], params["covariance"]
    [eigenvalues, eigenvectors] = np.linalg.eig(S)
    
    arg_eigen_major = np.argmax(abs(eigenvalues))
    arg_eigen_minor = max(1-arg_eigen_major, 0)
    lambda_sqrt_ = np.sqrt(eigenvalues)

    # plot contour
    for i in range(1, n_contours+1):
        ellipse = Ellipse(xy=(mean[0], mean[1]), 
            width=np.abs(lambda_sqrt_[arg_eigen_major])*2*i, height=np.abs(lambda_sqrt_[arg_eigen_minor])*2*i,
            angle=np.rad2deg(np.arctan(-eigenvectors[arg_eigen_major][1]/eigenvectors[arg_eigen_major][0])),
            edgecolor=color, linestyle = "--", fc='None', 
            lw=2, label="{} contour [{}]".format(label, i))
        
        ax = plt.gca()
        ax.add_patch(ellipse)

    # plot axis
    if plot_axis:
        plt.plot([mean[0], mean[0]+eigenvectors[arg_eigen_major][0]], 
                [mean[1], mean[1]-eigenvectors[arg_eigen_major][1]], 'black' )
        plt.plot([mean[0], mean[0]-eigenvectors[arg_eigen_minor][0]], 
                [mean[1], mean[1]+eigenvectors[arg_eigen_minor][1]], 'blue' )

def plot_decision_boundary_2d(
    params_dict,
    probability_model,
    x1lim      = (0, 500),
    x2lim      = (0, 500),
    N_steps    = 500,
    view_ang   = (45,45),
    merge_boundaries = True,
):
    ### Compute Z-maps:
    # Create grid and multivariate normal
    Nx, Ny = N_steps, N_steps
    x = np.linspace(x1lim[0],x1lim[1],Nx)
    y = np.linspace(x2lim[0],x2lim[1],Ny)
    XY = np.concatenate(np.transpose(np.meshgrid(x,y)), axis=0)
    Z_maps = {}
    for name_, params_ in params_dict.items():
        Z_maps[name_] = probability_model(x=XY, params=params_)
    
    ### Decision Boundary:
    dict_of_bnds = {}
    if merge_boundaries:
        classifier_occupancy_map = np.zeros(len(XY[:,0]))
        max_prob_map = np.zeros(len(XY[:,0]))
        class_index = 1 # 0: unknown (not classified)
        
        for name_, map_ in Z_maps.items():
            update_indices = np.where(map_ > max_prob_map)
            classifier_occupancy_map[update_indices] = class_index
            # update max
            max_prob_map = np.maximum(max_prob_map, map_)
            class_index += 1
        
        dict_of_bnds["combined"] = np.reshape(classifier_occupancy_map, (Nx, Ny))
    
    XX = np.reshape(XY[:,0], (Nx, Ny))
    YY = np.reshape(XY[:,1], (Nx, Ny))
    return dict_of_bnds, XX, YY


def output_2d_distribution(
    data_dict,
    params_dict,
    probability_model = None,
    figsize    = (12,6),
    OUT_DIR    = "",
    tag        = "",
    n_contours = 1,
    x1lim      = (0, 500),
    x2lim      = (0, 500),
    N_steps    = 500,
    verbose    = False,
):
    COLOR_TABLE = ["#b74f6fff","#628395ff","#dfd5a5ff","#dbad6aff","#cf995fff"]
    i = 1 # 0: not classified
    
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # plot contour:
    if probability_model is not None:
        N_unique = len(data_dict)
        dict_of_bnds, XX, YY = plot_decision_boundary_2d(
            params_dict          = params_dict,
            probability_model    = probability_model, 
            x1lim = x1lim, x2lim = x2lim, N_steps = N_steps,
        )
        cnt_fill = plt.contourf(XX, YY, dict_of_bnds["combined"], alpha=0.3, \
            levels=range(N_unique+1), colors=COLOR_TABLE[1:])
        cnt = ax.contour(XX, YY, dict_of_bnds["combined"], alpha=0.7, \
            levels=range(N_unique+1), colors="#9d0445")

    # plot data
    for name_, data_ in data_dict.items():
        plt.scatter(x=data_[:,0], y=data_[:,1], label=name_, c=COLOR_TABLE[i]) 
        if params_dict is not None and "gaussian" in params_dict[name_]["method"]:
            plot_probabilistic_contour(params=params_dict[name_], label=name_, \
                color=COLOR_TABLE[i], n_contours=n_contours)
        i += 1


    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_aspect(1)
    plt.grid()
    plt.title("2D Data Distribution [{}]".format(tag))
    fig.savefig("{}/data_distribution_{}.png".format(OUT_DIR, tag), bbox_inches = 'tight')
    if verbose:
        plt.show()
    plt.close(fig)
    return fig

def output_probabilities_in_3d(
    params_dict,
    probability_model,
    OUT_DIR    = "",
    tag        = "",
    figsize    = (12,12),
    x1lim      = (0, 500),
    x2lim      = (0, 500),
    N_steps    = 500,
    view_ang   = (45,45),
    verbose    = False,
):
    #Create grid and multivariate normal
    Nx, Ny = N_steps, N_steps
    x = np.linspace(x1lim[0],x1lim[1],Nx)
    y = np.linspace(x2lim[0],x2lim[1],Ny)
    XY = np.concatenate(np.transpose(np.meshgrid(x,y)), axis=0)

    #Make a 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    
    Z_max = np.zeros(len(XY[:,0]))
    for name_, params_ in params_dict.items():
        Z = probability_model(x=XY, params=params_)
        Z_max = np.maximum(Z, Z_max)
    
    # plot 3d mountains
    xx = np.reshape(XY[:,0], (Nx, Ny))
    yy = np.reshape(XY[:,1], (Nx, Ny))
    zz = np.reshape(Z_max, (Nx, Ny))
    ax.plot_surface(xx,yy,zz, cmap='viridis',linewidth=0, label=name_)

    ax.view_init(view_ang[0], view_ang[1]) 
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    plt.title("Estimated Probabilistic Distribution [{}]".format(tag))
    fig.savefig("{}/estimated_prob_distribution_{}.png".format(OUT_DIR, tag), bbox_inches = 'tight')
    if verbose:
        plt.show()
    plt.close(fig)
    return fig

#### CS 480 - Kaggle:
def bar_plot(data, variable):
    var =data[variable]
    varValue = var.value_counts()
    plt.figure(figsize=(15,7))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    
    plt.show()
    # print("{}: \n {}".format(variable,varValue))