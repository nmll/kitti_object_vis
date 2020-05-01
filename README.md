# KITTI Object data transformation and visualization
## Dataset

Download the data (calib, image\_2, label\_2, velodyne) from [Kitti Object trackin Dataset](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and place it in your data folder at `kitti/object`
before use this code,you need to convert kittitrackingdataset to kittidetection datasets format.Follow this：https://github.com/nmll/tools
after converting dataset format,change the filename 'object_tracking' to 'object'
The folder structure is as following:
```
./data
   object
      testing
        calib
        image_2	
        velodyne
      training
        calib
        image_2
        label_2
        velodyne
```
## Requirements

  - opencv
```
conda install opencv -c menpo
```
  - mayavi, vtk, PIL

```
pip install mayavi vtk pillow
# or
conda install mayavi vtk pillow
```

## note
2020.5.1 can show 3Ddetection results and 3Dtracking results on image and show BEV detection/tracking/gt bbox on lidar BEV


This is forked from https://github.com/kuixu/kitti_object_vis by me,this is for kittitrackingdataset format visualization
you can change the fps at raw 28 in this code
you could read next raw usage firstly

show detection bbox/gt on lidar topview and 2dgt,3dgt on image

```
$ python kitti_object.py --show_lidar_topview_with_boxes --img_fov --const_box --vis --show_image_with_boxes -p --detectdir saresult_val
```
show detection bbox/gt on lidar topview only

```
$ python kitti_object.py --show_lidar_topview_with_boxes --img_fov --const_box --vis  -p --detectdir saresult_val
```
for example show detection bbox/gt on lidar topview and 2d3dgt on image with 0012group default is 0000group
```
$ python kitti_object.py --show_lidar_topview_with_boxes --img_fov --const_box --vis --show_image_with_boxes -p --detectdir saresult_val --group 0012
```

show detection and tracking results bbox/gt on lidar topview only

```
$ python kitti_object.py --show_lidar_topview_with_boxes --img_fov --const_box --vis -p --show_tracking --detectdir saresult_val --trackdir carsassd_tra_val
```
show detection and tracking results bbox/gt on image only    but if detection results have some problems ,maybe raise some problems,if you could solve this,please contect me for my email!

```
$ python kitti_object.py --img_fov --const_box --vis --show_image_with_boxes -p --show_tracking --detectdir saresult_val --trackdir carsassd_tra_val --group 0003
```



## other notes
```
--ind 100   represent begin from 100th framid   priority level is higner than group ,group is recommended more -- maybe no useful
--group     use only this one can assign which group 0000-0020 , all the framid is end to 8008,but only current group detection/tracking results can be saw
--detectdir use to read 3D detection results from ./results/, eg --detectdir saresult_val means read from ./results/saresult_val
can used on windows by anaconda
--show_tracking    show tracking results,now only show on lidarBEV
--trackdir  input tracking results (follow ab3d output format) from which dir from ./results/,default=None
```

####below this is origin usage,for reference
## Visualization

1. 3D boxes on LiDar point cloud in volumetric mode
2. 2D and 3D boxes on Camera image
3. 2D boxes on LiDar Birdview
4. LiDar data on Camera image


```shell
$ python kitti_object.py --help
usage: kitti_object.py [-h] [-d N] [-i N] [-p] [-s] [-l N] [-e N] [-r N]
                       [--gen_depth] [--vis] [--depth] [--img_fov]
                       [--const_box] [--save_depth] [--pc_label]
                       [--show_lidar_on_image] [--show_lidar_with_depth]
                       [--show_image_with_boxes]
                       [--show_lidar_topview_with_boxes]

KIITI Object Visualization

optional arguments:
  -h, --help            show this help message and exit
  -d N, --dir N         input (default: data/object)
  -i N, --ind N         input (default: data/object)
  -p, --pred            show predict results
  -s, --stat            stat the w/h/l of point cloud in gt bbox
  -l N, --lidar N       velodyne dir (default: velodyne)
  -e N, --depthdir N    depth dir (default: depth)
  -r N, --preddir N     predicted boxes (default: pred)
  --gen_depth           generate depth
  --vis                 show images
  --depth               load depth
  --img_fov             front view mapping
  --const_box           constraint box
  --save_depth          save depth into file
  --pc_label            5-verctor lidar, pc with label
  --show_lidar_on_image
                        project lidar on image
  --show_lidar_with_depth
                        --show_lidar, depth is supported
  --show_image_with_boxes
                        show lidar
  --show_lidar_topview_with_boxes
                        show lidar topview
  --split               use training split or testing split (default: training)

```

```shell
$ python kitti_object.py
```
Specific your own folder,
```shell
$ python kitti_object.py -d /path/to/kitti/object
```

Show LiDAR only
```
$ python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis
```

Show LiDAR and image
```
$ python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis --show_image_with_boxes
```

Show LiDAR and image with specific index
```
$ python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis --show_image_with_boxes --ind 100 
```

Show LiDAR with label (5 vector)
```
$ python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis --pc_label
```

## Demo

#### 2D, 3D boxes and LiDar data on Camera image
<img src="./imgs/rgb.png" alt="2D, 3D boxes LiDar data on Camera image" align="center" />
<img src="./imgs/lidar-label.png" alt="boxes with class label" align="center" />
Credit: @yuanzhenxun

#### LiDar birdview and point cloud (3D)
<img src="./imgs/lidar.png" alt="LiDar point cloud and birdview" align="center" />

## Show Predicted Results

Firstly, map KITTI official formated results into data directory
```
./map_pred.sh /path/to/results
```

```python
python kitti_object.py -p
```
<img src="./imgs/pred.png" alt="Show Predicted Results" align="center" />


## Acknowlegement

Code is mainly from [f-pointnet](https://github.com/charlesq34/frustum-pointnets) and [MV3D](https://github.com/bostondiditeam/MV3D)
