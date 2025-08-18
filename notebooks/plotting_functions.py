# %%
# Visualization libraries


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker


# We're going to use plotly here, so we need to import it
import plotly.io as pio
import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go


# Adjust Basic Look of All Plots
## Set up Plotting
plt.rcParams['figure.figsize'] = [6, 4]  # Set a default figure size for the notebook
plt.rcParams['figure.dpi'] = 150  # Set default resolution for inline figures

# Set the default font size for axes labels, titles and ticks
plt.rcParams['axes.titlesize'] = 12  # Set the font size for axes titles
plt.rcParams['axes.labelsize'] = 12  # Set the font size for x and y labels
plt.rcParams['xtick.labelsize'] = 9 # Set the font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 9 # Set the font size for y-axis tick labels
plt.rcParams['font.size'] = 12 # Set the font size for the text in the figure (can affect legend)
plt.rcParams['legend.fontsize'] = 9  # Set the font size for legends

# set font to Avenir
plt.rcParams['font.family'] = 'Avenir'

# %%
def add_zebra_frame(ax, lw=2, segment_length=0.5, crs=ccrs.PlateCarree()):
    # Get the current extent of the map
    left, right, bot, top = ax.get_extent(crs=crs)
    # Check for valid extent
    if left == right or bot == top:
        return

    # Calculate the nearest 0 or 0.5 degree mark within the current extent
    left_start = left - left % segment_length
    bot_start = bot - bot % segment_length

    # Adjust the start if it does not align with the desired segment start
    if left % segment_length >= segment_length / 2:
        left_start += segment_length
    if bot % segment_length >= segment_length / 2:
        bot_start += segment_length

    # Extend the frame slightly beyond the map extent to ensure full coverage
    right_end = right + (segment_length - right % segment_length)
    top_end = top + (segment_length - top % segment_length)

    # Calculate the number of segments needed for each side
    num_segments_x = int(np.ceil((right_end - left_start) / segment_length))
    num_segments_y = int(np.ceil((top_end - bot_start) / segment_length))

    # Draw horizontal stripes at the top and bottom
    for i in range(num_segments_x):
        color = 'black' if (left_start + i * segment_length) % (2 * segment_length) == 0 else 'white'
        start_x = left_start + i * segment_length
        end_x = start_x + segment_length
        ax.hlines([bot, top], start_x, end_x, colors=color, linewidth=lw, transform=crs)

    # Draw vertical stripes on the left and right
    for j in range(num_segments_y):
        color = 'black' if (bot_start + j * segment_length) % (2 * segment_length) == 0 else 'white'
        start_y = bot_start + j * segment_length
        end_y = start_y + segment_length
        ax.vlines([left, right], start_y, end_y, colors=color, linewidth=lw, transform=crs)
# %%

def pacific_all_west_formatter(x, pos=None):
    # Convert to [0, 360)
    x = (x + 360) % 360
    deg_west = (180 - x) % 360
    if deg_west == 0:
        return "180°W"
    else:
        return f"{abs(deg_west):.0f}°W"
    

def plot_map(vmin, vmax, palette, xlims, ylims):
    """
    Plot a map of the magnitude of sea level change.

    Parameters:
    vmin (float): Minimum value for the color scale.
    vmax (float): Maximum value for the color scale.
    xlims (tuple): Tuple of min and max values for the x-axis limits.
    ylims (tuple): Tuple of min and max values for the y-axis limits.

    Returns:
    fig (matplotlib.figure.Figure): The matplotlib figure object.
    ax (matplotlib.axes._subplots.AxesSubplot): The matplotlib axes object.
    crs (cartopy.crs.Projection): The cartopy projection object.
    """
    # if xlims crosses the 180 meridian, set projection to central_longitude=180
    use_180 = False
    if xlims[1] > 180:
        crs = ccrs.PlateCarree(central_longitude=180)
        use_180 = True
        print("Using central_longitude=180 for projection")
    else:
        crs = ccrs.PlateCarree()
        print("Using default projection")

    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': crs})
    ax.set_extent([xlims[0], xlims[1], ylims[0], ylims[1]], crs=ccrs.PlateCarree())

    cmap = palette

    ax.coastlines()
    ax.add_feature(cfeature.LAND, color='lightgrey')


    gl = ax.gridlines(draw_labels=True, linestyle=':', color='black', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    return fig, ax, crs

# %%
def plot_map_grid(vmin, vmax, xlims, ylims, nrows, ncols):
    """
    Plot a map of the magnitude of sea level change.

    Parameters:
    vmin (float): Minimum value for the color scale.
    vmax (float): Maximum value for the color scale.
    xlims (tuple): Tuple of min and max values for the x-axis limits.
    ylims (tuple): Tuple of min and max values for the y-axis limits.

    Returns:
    fig (matplotlib.figure.Figure): The matplotlib figure object.
    ax (matplotlib.axes._subplots.AxesSubplot): The matplotlib axes object.
    crs (cartopy.crs.Projection): The cartopy projection object.
    cmap (matplotlib.colors.Colormap): The colormap used for the plot.
    """
    crs = ccrs.PlateCarree()
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10), subplot_kw={'projection': crs})
    for ax in axs.flat:
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
    
        ax.coastlines()
        ax.add_feature(cfeature.LAND, color='lightgrey')

    return fig, axs, crs

# %%
def plot_zebra_frame(ax, lw=5, segment_length=2, crs=ccrs.PlateCarree()):
    """
    Plot a zebra frame on the given axes.

    Parameters:
    - ax: The axes object on which to plot the zebra frame.
    - lw: The line width of the zebra frame. Default is 5.
    - segment_length: The length of each segment in the zebra frame. Default is 2.
    - crs: The coordinate reference system of the axes. Default is ccrs.PlateCarree().
    """
    # Call the function to add the zebra frame

    add_zebra_frame(ax=ax, lw=lw, segment_length=segment_length, crs=crs)
    # add map grid
    gl = ax.gridlines(draw_labels=True, linestyle=':', color='black',
                      alpha=0.5,xlocs=ax.get_xticks(),ylocs=ax.get_yticks())
    #remove labels from top and right axes
    gl.top_labels = False
    gl.right_labels = False

# %%
# def plot_thin_map_hawaii(labelStations=False):
#     import cartopy.crs as ccrs
#     import cartopy.feature as cfeature

#     # Create the plot
#     fig = plt.figure(figsize=(10, 7), constrained_layout=True)
#     ax = plt.axes(projection=ccrs.PlateCarree())

#     # Add features to the map
#     ax.add_feature(cfeature.LAND, facecolor='lightgray')
#     ax.add_feature(cfeature.COASTLINE, linewidth=0.25)
#     ax.add_feature(cfeature.BORDERS, linestyle=':')

#     gl = ax.gridlines(draw_labels=True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
#     gl.xlabel_style = {'size': 8}
#     gl.ylabel_style = {'size': 8}

#     # Set the extent to focus on Hawaii and surrounding areas
#     ax.set_extent([-179, -153, 15, 30])  # Adjust to focus on Hawaii

#     station_info = get_stationinfo()
#     station_label = {}

#     if labelStations:
#         # Add labels to the stations
#         for i, row in station_info.iterrows():
#             ax.scatter(row['lon'], row['lat'], color='black', s=10, transform=ccrs.PlateCarree())
#             station_label[i] = ax.text(row['lon'] + row['offsetlon'], row['lat'] + row['offsetlat'], row['station_name'],
#                     ha=row['ha'], va='center', transform=ccrs.PlateCarree(), fontsize=8, color=row['fontcolor'])

#     return fig, ax, station_info, station_label



