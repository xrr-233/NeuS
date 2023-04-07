 

# Training NeuS Using Your Custom Data

For dataset captured by hand, we use COLMAP SfM to transfer the dataset to the form that NeuS can accept.

**Step 1. Run COLMAP SfM**

Run  commands

```
cd colmap_preprocess
python imgs2poses.py ${data_dir}
```

After running the commands above, a sparse point cloud is saved in `${data_dir}/sparse_points.ply`.

**Step 2. Define the region of interest (Optional)**

The raw sparse point cloud may be noisy and may not be appropriate to define a region of interest (The white frame indicates the bounding box of the point cloud):

![raw_sparse_points](./static/raw_sparse_points.png)

And you may need to clean it by yourself (here we use Meshlab to clean it manually). After cleaning:

![interest_sparse_points](./static/interest_sparse_points.png)

Save it as `${data_dir}/sparse_points_interest.ply`.

Then run the commands:

```
python gen_cameras.py ${data_dir}
```

Then the preprocessed data can be found in `${data_dir}/preprocessed`.

### Notes

Here we just use the image without undistortion in the second option. To get better results, you may need to undistort your images in advance.



## Acknowledgement

The python scripts to run COLMAP SfM are heavily borrowed from LLFF: https://github.com/Fyusion/LLFF

