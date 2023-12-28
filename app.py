from flask import Flask, render_template, request, jsonify, send_file
import open3d as o3d
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import matplotlib
import tempfile
matplotlib.use('WebAgg')

from predcode import main

def divide_point_cloud(xyz):
   ranges = np.ptp(xyz, axis=0)
   min_range_axis = np.argmin(ranges)

   points_set1 = xyz[xyz[:, min_range_axis] < np.median(xyz[:, min_range_axis])]
   points_set2 = xyz[xyz[:, min_range_axis] >= np.median(xyz[:, min_range_axis])]

   xyz[:, min_range_axis] += 5.0

   return xyz, points_set1, points_set2

def calculate_area(xyz, color):
   ranges = np.ptp(xyz, axis=0)
   area = np.prod(np.sort(ranges)[-2:])

   pcd_model = o3d.geometry.PointCloud()
   pcd_model.points = o3d.utility.Vector3dVector(xyz)
   pcd_model.paint_uniform_color(color)
   bound_box = pcd_model.get_axis_aligned_bounding_box()
   # bound_box = pcd_model.get_oriented_bounding_box()
   bound_box.color = (0, 0, 0)

   return area, pcd_model, bound_box

class Dataform():

   # model = None
   color_list = None
   segment_plane = None
   colors = None
   org_seg = None
   bound_box = None
   save_pcd_path = None
   
data = Dataform() 

app = Flask(__name__,template_folder='template') 
UPLOAD_FOLDER = './pcdfiles'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

@app.route('/')  
def upload():  
   return render_template("file_upload_point_cloud.html")  
     
@app.route('/visualization', methods = ['POST',"GET"]) 
def visualization():  
   if request.method == 'POST':  
      
      f = request.files['file'] 

      file_str = f.filename.split(".")[-1]
      f.filename = f"PCDmodel.{file_str}"
      f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
      point_cloud_file_path = os.path.join(app.config['UPLOAD_FOLDER'],f.filename) #

      segment_planes, seg_color_name, seg_color_code = main(point_cloud_file_path)

      # save_pcd_path = os.path.join(app.config["UPLOAD_FOLDER"], "output.ply")
      # o3d.io.write_point_cloud(save_pcd_path, modelPCD)
      # data.save_pcd_path = save_pcd_path

      data.segment_plane = segment_planes
      data.color_list = seg_color_name
      data.colors = seg_color_code

      return render_template("visualization.html", options=data.color_list)#, dataframe=df_string) #, name = f.filename, image_fig = point_cloud_image_path)



@app.route('/result', methods= ['POST']) 
def result():

   # Use request.get_json() to retrieve JSON data
   formData = request.get_json()

   # Extract the 'selected_colors' key from the JSON data
   selected_colors = formData.get('selected_option', [])

   # print("Selected Colors:", selected_colors)
   idx = [data.color_list.index(i) for i in selected_colors] #data.color_list.index(selected_options)
   # print(idx)
   # selected_color_name = selected_colors
   selected_color_code = [data.colors[i] for i in idx] # data.colors[idx]
   data_points = [data.segment_plane[i] for i in idx] #  data.segment_plane[idx]

   pcd_model = o3d.geometry.PointCloud()
   for k in range(len(data_points)):
      pcd_model1 = o3d.geometry.PointCloud()
      pcd_model1.points = o3d.utility.Vector3dVector(data_points[k])
      pcd_model1.paint_uniform_color(selected_color_code[k])
      pcd_model += pcd_model1
   
   data.org_seg = pcd_model

   # selected_options = request.form['selected_option']
   # selected_options = request.form.getlist('selected_option')
   # print("Selected Colors:", selected_options)

   # idx = [data.color_list.index(i) for i in selected_options] #data.color_list.index(selected_options)
   # selected_color_name = data.color_list[idx]
   # selected_color_code = data.colors[idx]
   # data_points = data.segment_plane[idx]

   # if selected_color_name == "grey(remaining)":

   #    data.org_seg = data_points
   #    data.bound_box = []

   #    print_string = """
   #    Showing remaining point cloud data which 
   #    does NOT take part in segmentation"""
   
   # elif selected_color_name == "segmentation+remaining":
   #    data.org_seg = data_points
   #    data.bound_box = []

   #    print_string = """
   #    Showing full segmentation+remaining result"""
   
   # else:
   #    data_points_new, points_set1, points_set2 = divide_point_cloud(data_points)

   #    area, org_seg, bound_box = calculate_area(data_points_new, color=selected_color_code)
   #    area1, org_seg1, bound_box1 = calculate_area(points_set1, color=[0.33,0.42,0.18])
   #    area2, org_seg2, bound_box2 = calculate_area(points_set2, color=[1,0.51,0.98])

   #    print_string = f"""
   #    Area of wall  {selected_color_name}: {area:.4f}
   #    Area of wall olivegreen: {area1:.4f}
   #    Area of wall     orchid: {area2:.4f}"""
   #    org_seg  = org_seg + org_seg1 + org_seg2
   #    data.org_seg = org_seg
   #    data.bound_box = [bound_box, bound_box1, bound_box2]

   # print_string = "do something"
   
   # return jsonify({'print_string': print_string})
   return render_template('visualization.html')

@app.route('/visualize1/', methods=['GET']) 
def visualize1_route():
   # print("visualize1_route")
   # o3d.visualization.draw_geometries([data.org_seg] + [bb for bb in data.bound_box if (len(data.bound_box) > 0)], window_name=f'wall_segment', width = 1800, height = 1000, left = 10)
   coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
   o3d.visualization.draw_geometries([data.org_seg, coord_frame], window_name=f'wall_segment, X-Red, Y-Green, Z-Blue', width = 1800, height = 1000, left = 10)
   return render_template('visualization.html')

@app.route('/download_point_cloud', methods=['GET'])
def download_point_cloud():
   file_format = request.args.get('file_format')
   if file_format == 'txt':
      # Save the point cloud to a temporary XYZ file
      points = np.asarray(data.org_seg.points)
      colors = np.asarray(data.org_seg.colors)
      points_with_colors = np.hstack((points, colors))

      temp_xyz_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
      np.savetxt(temp_xyz_file.name, points_with_colors, fmt='%.4f', delimiter=' ')
      temp_xyz_file.close()


      # Serve the temporary XYZ file for download
      return send_file(temp_xyz_file.name, as_attachment=True, download_name='point_cloud.txt')

   elif file_format == 'ply':
      # Save the point cloud to a temporary PLY file
      temp_ply_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
      o3d.io.write_point_cloud(temp_ply_file.name, data.org_seg)
      temp_ply_file.close()

      # Serve the temporary PLY file for download
      return send_file(temp_ply_file.name, as_attachment=True, download_name='point_cloud.ply')

   #  return send_file(data.save_pcd_path, as_attachment=True)

@app.route('/export_excel', methods=['GET'])
def export_excel():

   # Create an empty DataFrame
   df = pd.DataFrame(columns=['Color_name', 'Color_code', 'Area', 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'])
   df_data = []
   for i in range(len(data.segment_plane)-1):
      data_points = data.segment_plane[i]

      ranges = np.ptp(data_points, axis=0)
      min_index = np.argmin(ranges)  # get minimum range value index
      # delete the minimum range coordinate value
      plot_data = np.delete(data_points, min_index, 1)
      x_range = [plot_data[:, 0].min(), plot_data[:, 0].max()]
      y_range = [plot_data[:, 1].min(), plot_data[:, 1].max()]

      fig = plt.figure()
      plt.scatter(plot_data[:, 0], plot_data[:, 1], s=1, c='skyblue', marker='o')
      plt.xlabel('X')
      plt.ylabel('Y')
      plt.xlim(x_range)
      plt.ylim(y_range)

      plt.savefig('static\images\contour_image.png')
      plt.close()

      image = cv2.imread("Extra\contour_image.png") 

      # Convert to grayscale 
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
      # Blur the image 
      blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
      thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
      # Detect edges 
      edges = cv2.Canny(blurred, 50, 150) 
      # Find contours 
      contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
      # Filter contours 
      rects = [] 
      for contour in contours: 
         # Approximate the contour to a polygon 
         polygon = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True) 
         
         # Check if the polygon has 4 sides and the aspect ratio is close to 1 
         if len(polygon) == 4 and abs(1 - cv2.contourArea(polygon) / (cv2.boundingRect(polygon)[2] * cv2.boundingRect(polygon)[3])) < 0.1: 
            rects.append(polygon) 

      # Find the largest rectangle
      largest_rect = max(rects, key=cv2.contourArea)
      # Get the bounding box of the largest rectangle
      x, y, w, h = cv2.boundingRect(largest_rect)

      cropped_image = thresh[y:y+h-2, x:x+w-2]
      invert = cv2.bitwise_not(cropped_image)
      num_0 = np.sum(invert == 0)
      num_255 = np.sum(invert == 255)

      ratio = num_0 / (num_255 + num_0)
      area = ratio * (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])

      min_values = np.round(data_points.min(axis=0), 3)
      max_values = np.round(data_points.max(axis=0), 3)


      # ranges = np.round(np.ptp(data_points, axis=0), 2)
      # sort_ranges = np.sort(ranges)
      # area = np.prod(sort_ranges[-2:])

      df_data.append({'Color_name': data.color_list[i], 'Color_code':data.colors[i], 'Area':np.round(area,3), 
                      'x_min':min_values[0], 'x_max':max_values[0], 'y_min':min_values[1], 'y_max':max_values[1], 
                      'z_min':min_values[2], 'z_max':max_values[2]})
   
   
   df = pd.concat([df, pd.DataFrame(df_data)], ignore_index=True)
   excel_file_path = 'pcdfiles/Area_color_segments.xlsx'
   df.to_excel(excel_file_path)
   

   # Serve the temporary PLY file for download
   return send_file(excel_file_path, as_attachment=True, download_name='Area_color_segments.xlsx')

   #  return send_file(data.save_pcd_path, as_attachment=True)

@app.route('/visualize2') 
def visualize2():
   model_corners = o3d.geometry.PointCloud()
   bound_boxs = []
   for i in range(len(data.segment_plane)-1):

      data_points = data.segment_plane[i]
      pcd_model = o3d.geometry.PointCloud()
      pcd_model.points = o3d.utility.Vector3dVector(data_points)

      bound_box = pcd_model.get_axis_aligned_bounding_box()
      bound_box.color = data.colors[i]

      points = np.asarray(bound_box.get_box_points())
      pcd1 = o3d.geometry.PointCloud()
      pcd1.points = o3d.utility.Vector3dVector(points)
      pcd1.paint_uniform_color([0,0,0])

      bound_boxs.append(bound_box) 
      model_corners += pcd1
   o3d.visualization.draw_geometries([model_corners]+ [box for box in bound_boxs], window_name='Bounding boxes for each Wall', width = 1800, height = 1000, left = 10)
   return render_template('visualization.html')


# @app.route('/visualize2') 
# def visualize2():
#    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
#    o3d.visualization.draw_geometries([data.model, coord_frame], window_name='Segmentation', width = 1800, height = 1000, left = 10)
#    return render_template('visualization.html')


if __name__ == '__main__':  
   app.run(host='0.0.0.0', port=8080)  
      
# if __name__ == '__main__':  
#   app.run()#debug = True)

# AWS : https://www.youtube.com/watch?v=_rwNTY5Mn40
# sudo apt-get update && sudo apt-get install pip

   
# OSError: libGL.so.1: cannot open shared object file: No such file or directory
# To solve this:
# apt-get install libgl1-mesa-glx
