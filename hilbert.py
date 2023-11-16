#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import time
from math import log
import numpy as np
from scipy import interpolate
import hashlib
import math 
from PIL import Image, ImageDraw, ImageFont
from colorsys import hsv_to_rgb
import cmath
from math import cos, sin, radians

def hilbert_curve_order_curve(n):
    """
    This is a function that traverse the hilbert curver of ordre n
    At each point i, on the hilbert curve, it draws the ith order hilbert curve
    this function returns a list of lists of points
    """
    # Call hilber_curve_iter(n) to get the main hilber curve, then at each point on the curve
    # call hilbert_curve_iter(i) to get the ith order hilbert curve
    # then translate the points of the ith order curve to the point on the main curve
    # make sure the ith order curve is scaled down to fit in the main curve
    nth_order_curve = hilbert_curve_iter(n)
    points = []
    for i in range(len(nth_order_curve)):
        # take the base 2 log of i so that the curve detail doesnt become too big
        if i == 0:
          curve_num = 1
        else:
          curve_num = int(log(i, 3)) + 1
        ith_order_curve = hilbert_curve_iter(curve_num)
        #scale the ith order curve to fit in the main curve
        for point in ith_order_curve:
            point[0] /= (1 << n)
            point[1] /= (1 << n)
        # translate the points of the ith order curve to the point on the main curve
        for point in ith_order_curve:
            point[0] += nth_order_curve[i][0]
            point[1] += nth_order_curve[i][1]
        points += ith_order_curve
    return points

def hilbert_curve_continous(n):
    """
    This function takes in a float n and returns a list of lists of points
    """
    # calculate floor(n) hilbert 
    # calculate ceil(n) hilbert
    # if floor(n) = k , then the curve will have 2^k points and ceil(n) will have 2^(k+1) points
    # we need to map find the equivalent points on the floor(n) curve for the ceil(n) curve
    # to do this we divide 2^(k+1) by 2^k to get 2
    # this means there will be 2 points in between any two points
    # to get the location of these points, we get horizontal and vertical distance between the two points
    # then we multiply the distance by 1/3, 2/3 and add it to the first point to get the location of the points
    # n = n / 2

    if n < 2:
        order_interval = 1
    else:
        order_interval = 1

    lower_order = int(int(n) / order_interval) * order_interval
    higher_order = lower_order + order_interval

    # print(f'lower order: {lower_order}')
    # print(f'higher order: {higher_order}')
    # print(f'n: {n}')

    

    floor_points = hilbert_curve_iter(int(lower_order))
    ceil_points = hilbert_curve_iter(int(higher_order))
    points = []

    #print length
    p = int(len(ceil_points) / len(floor_points)) + 1 # number of points inbetween two points

    for i in range(len(floor_points) - 1):
        # get the horizontal and vertical distance between the two points
        x_distance = floor_points[i + 1][0] - floor_points[i][0]
        y_distance = floor_points[i + 1][1] - floor_points[i][1]
        # get the location of the intermediate points
        for j in range(0,p):
            points.append([floor_points[i][0] + x_distance * j/p, floor_points[i][1] + y_distance * j/p])
    points.append(floor_points[-1])

    # now we will linearly interpolate between these points and the equivalent one on the ceil curve
    # We will use the fractional part of n to determine the weight of the interpolation
    # if the fractional part if 0.5, we will lerp halfway between the two points
    # if the fractional part is 0.25, we will lerp 1/4 of the way between the two points

    floor_points_len = len(points)
    ceil_points_len = len(ceil_points)

    scale = ceil_points_len / floor_points_len

    fractional_part = ((n / float(order_interval)) % 1.0) 




    for j in range(len(points)):
        
        i = int(j * scale)
        # get the horizontal and vertical distance between the two points
        x_distance = ceil_points[i][0] - points[j][0]
        y_distance = ceil_points[i][1] - points[j][1]
        # get the location of the intermediate points
        first_point = points[j]
        second_point = [first_point[0] + x_distance * fractional_part, first_point[1] + y_distance * fractional_part]
        points[j] = second_point
    
    return points






def hilbert_curve_iter(n):
    points = []
    for i in range(1 << (n << 1)):
        x = y = 0
        t = i
        s = 1
        while s < (1 << n):
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s <<= 1
        points.append([x, y])
    
    if n == 0:
        points.append([0.5, 0])

    # normalize the points so that they are between 0 and 1, with the tallest points at 1 and lowest points at 0
    # get the max and min x and y values
    max_x = max(points, key=lambda x: x[0])[0]
    min_x = min(points, key=lambda x: x[0])[0]
    max_y = max(points, key=lambda x: x[1])[1]
    min_y = min(points, key=lambda x: x[1])[1]

    #for order 0, add two points to the list, of the same value at 0.5

    
    # handle division by 0 cases
    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1
    # normalize the points
    for point in points:
        point[0] = (point[0] - min_x) / (max_x - min_x)
        point[1] = (point[1] - min_y) / (max_y - min_y)
    return points

def hilbert_curve_memo(n, memo={}):
    if n == 0:
        return [[0.5, 0.5]]

    if n in memo:
        return memo[n]

    prev = hilbert_curve_memo(n - 1, memo)
    new_curve = []

    for point in prev:
        new_curve.append([point[1] / 2, point[0] / 2])  # bottom left

    for point in prev:
        new_curve.append([point[0] / 2, point[1] / 2 + 0.5])  # top left

    for point in prev:
        new_curve.append([point[0] / 2 + 0.5, point[1] / 2 + 0.5])  # top right

    for point in prev[::-1]:
        new_curve.append([1 - point[1] / 2, point[0] / 2])  # bottom right

    memo[n] = new_curve
    return new_curve

def hilbert_curve(n):
    """
    This function takes a positive integer n and returns a list of lists
    Each list in the list is a coordinate pair (x,y) on the Hilbert C
    curve of order n.
    Each point is a float value between 0 and 1
    """
    if n == 0:
        return [[0.5,0.5]]
    
    # leave the top left and top right corners in the same orientation, but make the half the sixe
    # rotate the bottome left corner 90 degrees clockwise
    # rotate the bottom right corner 90 degrees counter clockwise

    prev = hilbert_curve(n-1)
    new_curve = []

    # bottom left
    for point in prev:
        new_curve.append([point[1]/2, point[0]/2])
    # top left
    for point in prev:
        new_curve.append([point[0]/2, point[1]/2 + 0.5])
    # top right
    for point in prev:
        new_curve.append([point[0]/2 + 0.5, point[1]/2 + 0.5])
    # bottom right
    # add the bottome right points in reverse order
    for point in prev[::-1]:
        new_curve.append([1 - point[1]/2, point[0]/2])
    
    return new_curve

# use mat plot lib to plot the curve in red

def plot_line(points):
    """
    This function takes in a list of lists of points and plots them
    """
    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x,y, 'r')
    plt.show()

def plot_curvy_line(points):
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # Compute b-spline representation of the curve
    tck, _ = interpolate.splprep([x,y], s=0)

    # Generate new points on the curve
    new_points = np.linspace(0, 1, 1000)
    new_x, new_y = interpolate.splev(new_points, tck)

    # Plot
    plt.figure()
    plt.plot(new_x, new_y)
    plt.axis('equal') # Ensure that the axes are scaled equally
    plt.show()

def plot_line_gradient(points):
    """
    This plots a line with a gradient transitioning between two colours as it plot
    """
    color1 = 'red'
    color2 = 'blue'
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', [color1, color2])

    fig, ax = plt.subplots()

    for i in range(len(points) - 1):
        x_values = [points[i][0], points[i + 1][0]]
        y_values = [points[i][1], points[i + 1][1]]
        line_color = cmap(i / (len(points) - 2))
        ax.plot(x_values, y_values, color=line_color)

    plt.show()

def plot_line_with_points(points):
    """
    This function takes in a list of lists of points and plots them
    """
    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x,y, 'r')
    plt.scatter(x,y, s=20, c='b')
    plt.show()

def draw_segments_on_canvas(canvas, points, colors, width, height, background_color=(0, 0, 0)):
    
    line_width = 6

    draw = ImageDraw.Draw(canvas)
    # draw.rectangle((0, 0, width, height), fill=background_color)

    smallest = max(width, height) # ; )



    scaled_points = [(int(p[0] * smallest), int(p[1] * smallest)) for p in points]

    # move all points to the center of the canvas
    scaled_points = [(p[0] + (width - smallest) / 2, p[1] + (height - smallest) / 2) for p in scaled_points]

    for i in range(len(scaled_points) - 1):
        start_point = scaled_points[i]
        end_point = scaled_points[i + 1]
        color = tuple(colors[i])

        #convert color from float to int between 0 and 255
        color = tuple((np.array(color) * 255).astype(int))

        # add extension to line that is equal to the width of the line
        # this is to prevent bad corners
        # use linear algebra to do this
        # if the line goes out of bounds with the extension, then clip it

        # get the vector from start to end
        vector = np.array(end_point) - np.array(start_point)

        # deal with the case where the vector is 0 by not add the extension
        if np.linalg.norm(vector) == 0:
            draw.line([start_point, end_point], fill=color, width=line_width)
            continue
            
        # get the unit vector
        unit_vector = vector / np.linalg.norm(vector)
        # get the unit * the extension length
        extension_vector = unit_vector * (line_width * 0.25)

        # get the extended start and end points
        extended_start_point = start_point - extension_vector
        extended_end_point = end_point + extension_vector

        canvas_size = max(width, height)
        # clip the extended points
        extended_start_point = np.clip(extended_start_point, 0, canvas_size)
        extended_end_point = np.clip(extended_end_point, 0, canvas_size)

        # cast to tuple of ints
        extended_start_point = tuple(extended_start_point.astype(int))
        extended_end_point = tuple(extended_end_point.astype(int))


        draw.line([extended_start_point, extended_end_point], fill=color, width=line_width)
    
    return canvas

def hash_fn_val(value):
    x = hashlib.sha256(str(value).encode('utf-8')).hexdigest()
    y = int(x, 16)
    z = y / (2 ** 256)
    return z

def hash_fn(value):
    # map every value in an even way to a float between 0 and 1
    # make it seem random and unorderly, but dont use perlin noise
    # use a hash function
    hue = hash_fn_val(value)

    #make the saturation and value a reverse bell curve, so that we dont have many grey colors
    # we want values around the middle to be very unlikely
    hashed_val1 = hash_fn_val(value + 1)
    hashed_val2 = hash_fn_val(value + 2)
    saturation = 1.0#hashed_val1 * 0.25 + 0.75
    if hashed_val1 < 0.1:
        saturation = hashed_val1
    value = 1.0
    # if hashed_val2 < 0.2:
    #     value = hashed_val2
        

    # bell_1 = 1 - stats.norm.cdf(stats.norm.ppf(hashed_val1))
    # bell_2 = 1 - stats.norm.cdf(stats.norm.ppf(hashed_val2))
    # saturation = bell_1 * 0.25 + 0.75
    # value = bell_2 * 0.25 + 0.75
    # print(f'saturation: {saturation}')
    # print(f'value: {value}')

    #convert to rgb
    r, g, b = hsv_to_rgb(hue, saturation, value)
    return np.array([r, g, b])

def continuous_random_walk(value, step_size=1.0):
    lower_step = math.floor(value / step_size) * step_size
    upper_step = math.ceil(value / step_size) * step_size

    lower_point = hash_fn(lower_step)
    upper_point = hash_fn(upper_step)

    t = (value - lower_step) / (upper_step - lower_step)

    # Linearly interpolate between the two closest points
    return (1 - t) * lower_point + t * upper_point

def get_coordinate(value):
    value = value * 0.001
    return continuous_random_walk(value, 0.001)





def animate_curve(init_val, final_val, step, curve_func):
    """
    This function takes in the initial value, final value and step
    It then animates the curve from the initial value to the final value
    """
    # use FuncAnimation to animate the curve
    # the curve_func should take in a float and return a list of lists of points
    fig = plt.figure(figsize=(14, 7))  # For example, 8x8 inches

    # Add a subplot to the figure
    ax = fig.add_subplot(111)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=None, hspace=None)  # Adjust subplot parameters


    color_start = np.array([1.0, 0.0, 0.0])  # red
    color_end = np.array([0.0, 1.0, 0.0])  # green

    max_change = 0.1


    rotation = 0.0 # 0 - 1  float maps to 0 - 360 degrees

    resolution_scale = 2.6
    width, height = int(350 * resolution_scale), int(175 * resolution_scale)
    canvas = Image.new('RGB', (width, height), 'black')

    #choose a random starting value
    color_iter = time.time()

    size_iter = 0

    def animate(i):
        
        nonlocal color_start, color_end, canvas, rotation, color_iter, size_iter
        ax.cla()


        points = curve_func(i)

        spacing = 2.1

        #copy all the points add them to the list, but but each copy should be tiled around the center
        new_points = []
        for point in points:
            point = np.array(point)
            point = point - 0.5
            point[1] = -point[1]
            point[0] = -point[0]
            point = point + 0.5
            point = point.tolist()
            new_points.append([point[0], spacing - point[1]])
        for point in points:
            # move to the middle and flip over the y axis
            point = np.array(point)
            point = point - 0.5
            point[1] = -point[1]
            point = point + 0.5
            point = point.tolist()
            new_points.append(point)
        for point in points:
            # move to the middle and flip over the y axis
            point = np.array(point)
            point = point - 0.5
            point[1] = -point[1]
            point[0] = -point[0]
            point = point + 0.5
            point = point.tolist()
            new_points.append([spacing - (point[0]), point[1]])
        for point in points:
            # move to the middle and flip over the y axis
            point = np.array(point)
            point = point - 0.5
            point[1] = -point[1]
            point = point + 0.5
            point = point.tolist()
            new_points.append([spacing - point[0], spacing - point[1]])

            
        points = new_points

        #recenter
        for point in points:
            point[0] -= 0.5 + (spacing - 2) / 2
            point[1] -= 0.5 + (spacing - 2) / 2


        # rotate the points around 0.5, 0.5
        points = np.array(points)
        rotation_matrix = np.array([[np.cos(rotation * 2 * np.pi), -np.sin(rotation * 2 * np.pi)], [np.sin(rotation * 2 * np.pi), np.cos(rotation * 2 * np.pi)]])
        points = points - 0.5
        points = np.matmul(points, rotation_matrix)
        
        size_iter += 0.01
        freq_1 = abs(np.sin(size_iter * 0.7)) * 1.2
        freq_2 = np.sin(size_iter * 0.42) * 0.3
        freq_3 = np.sin(size_iter * 0.2) * 0.1
        size = freq_1 * 0.4 + 0.4 + freq_2 + freq_3
        # points = points * 0.2
        points = points * size * 0.4

        points = points + 0.5

        # move left and right with sine
        freq_1 = np.sin(size_iter * 0.1) * 0.1
        freq_2 = np.sin(size_iter * 0.5) * 0.12
        freq_3 = np.sin(size_iter * 4) * 0.06
        points = points + np.array([freq_1 + freq_2, freq_2 + freq_3])



        freq_1 = (np.sin(size_iter * 0.1) ** 3) * 0.004 * 0.5
        freq_2 = (np.sin(size_iter * 2) ** 3) * 0.002 * 0.5
        rotation = (rotation + freq_1 + freq_2  ) % 1


        saturated_color = get_coordinate(color_iter)
        # saturated_color = (0.0,0.0,0.0)
        unsaturated_color = get_coordinate(color_iter + 0.5)

        #based on coler iter, lerp between saturated and unsaturated color using a library for lerping
        #use the fractional part of the coler iter to determine the weight of the interpolation
        frational_part = color_iter % 1
        lerped_color_start = (1 - frational_part) * np.array(saturated_color) + frational_part * np.array(unsaturated_color)
        lerped_color_end = (1 - frational_part) * np.array(unsaturated_color) + frational_part * np.array(saturated_color)


        num_points = points.shape[0]
        color_iter += 0.005 #some random number between 0 and 1
        color_start = lerped_color_start
        color_end = lerped_color_end

        # Ensure the colors stay valid by clipping them between 0 and 1
        color_start = np.clip(color_start, 0, 1)
        color_end = np.clip(color_end, 0, 1)




        # instead of lerping between the colors, I want to have two colours that have been lerped between 
        # then I want to slide that gradient along the curve
        # I can do this by using color iter to determine at what point on the curve the most saturated and unsturated colors are
        # then I can lerp between the two colors based on the distance from the most saturated and unsaturated points
        # I can use the fractional part of the color iter to determine the weight of the interpolation

        saturated_location = color_iter % 1
        unsaturated_location = (color_iter + 0.5) % 1

        # get the lerped colors between the saturated and unsaturated colors, then flip them and add them to the list, so it goes in both directions

        # get the distance between the saturated and unsaturated locations
        distance = abs(saturated_location - unsaturated_location)
        # get the colors for the corrsponding points
        colors = []
        for j in range(num_points):
            # get the location of the point
            point_location = j / num_points
            # get the distance from the saturated and unsaturated locations
            # make sure the distance wraps around, so that 0.9 and 0.1 are close together
            saturated_distance = min(abs(point_location - saturated_location), 1 - abs(point_location - saturated_location))
            unsaturated_distance = min(abs(point_location - unsaturated_location), 1 - abs(point_location - unsaturated_location))
            # get the weight of the interpolation
            saturated_weight = 1 - (saturated_distance / distance)
            unsaturated_weight = 1 - (unsaturated_distance / distance)
            # add the lerped color to the list
            #use saturate_color and unsaturated_color
            # actually make the interpolation between the two colors squared, so that the colors are more saturated
            lerped_color = (saturated_weight ** 2) * np.array(saturated_color) + (unsaturated_weight ** 2) * np.array(unsaturated_color)
            colors.append(lerped_color)
        
        #convert colors to 2d array np
        colors = np.array(colors)
            
        canvas = draw_segments_on_canvas(canvas, points, colors, width, height)

        ax.imshow(canvas)
        ax.set_title(f"Hilbert curve of order {i}")
    
    num_frames = 100  # Adjust as necessary, reduced due to forward and reverse

    frame_reduction_constant = 2.0

    # For the forward frames
    forward_indices = np.linspace(0, 1, num_frames)
    forward_orders = init_val + (np.log(forward_indices + 1) / np.log(2)) ** frame_reduction_constant  * (final_val - init_val)

    # For the reverse frames, flip the forward_orders
    reverse_orders = np.flip(forward_orders)
    # reverse_orders = forward_orders

    # Combine the two arrays
    frames = np.concatenate((forward_orders, reverse_orders))
    # frames = forward_orders


    anim = FuncAnimation(fig, animate, frames=frames, interval=1, blit=False)
    plt.show()

  

    




def main():
    """
    This function takes in a positive integer n and plots the Hilbert C curve of order n
    """
    # n = float(input("Enter a positive integer: "))
    n = 3

    # get the current time
    # start_time = time.perf_counter()
    # points = hilbert_curve_continous(n)
    # end_time = time.perf_counter()
    # execution_time = end_time - start_time
    # print(f"Execution time: {execution_time} seconds")
    # plot_line_gradient(points)
    # plot_curvy_line(points)
    # plot_line_with_points(points)
    animate_curve(1, n, 0.2, hilbert_curve_continous)

if __name__ == "__main__":
    main()
