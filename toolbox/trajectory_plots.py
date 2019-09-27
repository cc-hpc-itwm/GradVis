import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D
from IPython.core.display import display, HTML
import json
import numpy as np

def plot3D(X, Y, Z, height=600, xlabel = "X", ylabel = "Y", zlabel = "Z", initialCamera = None):
    """
    Plots the loss landscape in an interactive way.

    Args:
        X, Y, Z: x,y,z-components of loss landscape.
        height: Starting value of the height.
        (x-,y-,z-)label: Label of the x-,y-,z-axis.
        initialCamera: Initial camera position.
    """

    options = {
        "width": "100%",
        "style": "surface",
        "showPerspective": True,
        "showGrid": True,
        "showShadow": False,
        "keepAspectRatio": True, #True for true ratio
        "height": str(height) + "px"
    }

    if initialCamera:
        options["cameraPosition"] = initialCamera

    data = [ {"x": X[y,x], "y": Y[y,x], "z": Z[y,x]} for y in range(X.shape[0]) for x in range(X.shape[1]) ]
    visCode = r"""
       <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" type="text/css" rel="stylesheet" />
       <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
       <div id="pos" style="top:0px;left:0px;position:absolute;"></div>
       <div id="visualization"></div>
       <script type="text/javascript">
        var data = new vis.DataSet();
        data.add(""" + json.dumps(data) + """);
        var options = """ + json.dumps(options) + """;
        var container = document.getElementById("visualization");
        var graph3d = new vis.Graph3d(container, data, options);
        graph3d.on("cameraPositionChange", function(evt)
        {
            elem = document.getElementById("pos");
            elem.innerHTML = "H: " + evt.horizontal + "<br>V: " + evt.vertical + "<br>D: " + evt.distance;
        });
       </script>
    """
    htmlCode = "<iframe srcdoc='"+visCode+"' width='100%' height='" + str(height) + "px' style='border:0;' scrolling='no'> </iframe>"
    display(HTML(htmlCode))


def plot_loss(X,Y,Z,path_x=[],path_y=[],path_z=[],filename="out_3D", height=10,degrees=210, is_log=False):
    """
    Creates a 3D-plot of the loss and with the trajectory taken.

    Args:
        X, Y, Z: x,y,z-components of loss landscape.
        path_(x,y,z): x,y,z-components of path values.
        filename: name of the plot that is saved.
        height: height of the landscape in degrees.
        degrees: turns the plot by the amount of degrees specified.
        is_log (boolean): plot the logarithmic loss landscape and trajectory.
    """
    if is_log:
        scale = lambda x: np.log(x)
    else:
        scale = lambda x: x

    fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, scale(Z),cmap="cool")
    if len(path_x)!=0 and len(path_y)!=0 and len(path_z)!=0:
        NPOINTS=len(path_x)
        cmap = matplotlib.cm.get_cmap('hot')
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=NPOINTS)
        colors = [cmap(normalize(value)) for value in range(NPOINTS)][::-1]
        ax.plot(path_x,path_y,scale(path_z),"k-",zorder=4)
        ax.scatter(path_x,path_y,scale(path_z),color=colors,marker="o")

        for k in range(NPOINTS):
            ax.plot([path_x[k]],[path_y[k]],scale(path_z[k]), markerfacecolor=colors[k], markeredgecolor=colors[k], marker='o', markersize=5, alpha=1,zorder=10)

        ax.plot([path_x[-1]],[path_y[-1]],scale(path_z[-1]), markerfacecolor='r', markeredgecolor='r', marker='X', markersize=10, alpha=0.6,zorder=11)
        ax.plot([path_x[0]],[path_y[0]],scale(path_z[0]), markerfacecolor='k', markeredgecolor='k', marker='v', markersize=10, alpha=0.6,zorder=11)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(height, degrees)
    fig_savename='%s.svg'%(filename)
    print('Saving figure %s'%fig_savename)
    fig.savefig(fig_savename, dpi=300, format='svg')
    plt.close(fig)



def contour_loss(X,Y,Z,path_x=[],path_y=[],labels=[], filename="out", is_log=False):
    """
    Creates a 2D-plot of the loss and with the trajectory taken.

    Args:
        X, Y, Z: x,y,z-components of loss landscape.
        path_(x,y,z): x,y,z-components of path values.
        filename: name of the plot that is saved.
        labels: [x,y]-label used in the plot.
        is_log (boolean): plot the logarithmic loss landscape and trajectory.
    """
    if is_log:
        scale = lambda x: np.log(x)
    else:
        scale = lambda x: x

    fig = plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k', frameon=False)
    CS = plt.contour(X, Y, scale(Z),100,zorder=0)

    if len(path_x)!=0 and len(path_y)!=0:
        NPOINTS=len(path_x)
        cmap = matplotlib.cm.get_cmap('hot')
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=NPOINTS)
        colors = [cmap(normalize(value)) for value in range(NPOINTS)]
        As0 = plt.plot(path_x,path_y, color="k",markersize=0, alpha=1,zorder=8)
        As = plt.scatter(path_x,path_y,color=colors[::-1],edgecolor='k',alpha=1,zorder=10)
        As1 = plt.plot(path_x[0],path_y[0], markerfacecolor='k', markeredgecolor='k', marker='v', markersize=10, alpha=1,zorder=10)
        As1 = plt.plot(path_x[-1],path_y[-1], markerfacecolor='r', markeredgecolor='r', marker='X', markersize=10, alpha=1,zorder=10)

    plt.clabel(CS, inline=1, fontsize=10)
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis='y',which='both',bottom=False,top=False,labelbottom=False)
    if len(labels)!= 0:
        plt.xlabel(str(labels[0]))
        plt.ylabel(str(labels[1]))

    fig_savename='%s.svg'%(filename)
    print('Saving figure %s'%fig_savename)
    fig.savefig(fig_savename, dpi=300, format='svg')
    plt.close(fig)


def plot_loss_2D(path_to_file,filename="out", is_log=False):
    """
    Wrapper for contour_loss function. Opens the .npz file created by the visualize function and creates a 2D plot.

    Args:
        path_to_file: path to the .npz file created by visualize.
        filename: name of the plot that is saved.
        is_log (boolean): plot the logarithmic loss landscape and trajectory.
    """
    outs = np.load(path_to_file)
    flag = outs["b"]
    outs = outs["a"]

    if flag == 1:
        contour_loss(outs[0][0],outs[0][1],outs[0][2],path_x=outs[1][0],path_y=outs[1][1], filename=filename, is_log=is_log)

    elif flag == 2:
        contour_loss(outs[0][0],outs[0][1],outs[0][2],path_x=outs[1][0],path_y=outs[1][1],labels=outs[2][0], filename=filename, is_log=is_log)

    elif flag == 3:
        contour_loss(outs[0][0],outs[0][1],outs[0][2], filename=filename, is_log=is_log)


def plot_loss_3D(path_to_file, filename="out_3D", height=10,degrees=210, is_log=False):
    """
    Wrapper for plot_loss function. Opens the .npz file created by the visualize function and creates a 3D plot.

    Args:
        path_to_file: path to the .npz file created by visualize.
        filename: name of the plot that is saved.
        height: height of the landscape in degrees.
        degrees: turns the plot by the amount of degrees specified.
        is_log (boolean): plot the logarithmic loss landscape and trajectory.
    """
    outs = np.load(path_to_file)
    flag = outs["b"]
    outs = outs["a"]

    if flag == 1:
        plot_loss(outs[0][0],outs[0][1],outs[0][2],path_x=outs[1][0],path_y=outs[1][1],path_z=outs[1][2], height=height,degrees=degrees, filename=filename, is_log=is_log)

    elif flag == 2:
        plot_loss(outs[0][0],outs[0][1],outs[0][2],path_x=outs[1][0],path_y=outs[1][1],path_z=outs[1][2], height=height,degrees=degrees, filename=filename, is_log=is_log)

    elif flag == 3:
        plot_loss(outs[0][0],outs[0][1],outs[0][2], filename=filename, height=height,degrees=degrees, is_log=is_log)