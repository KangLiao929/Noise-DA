For training and testing, your directory structure should look like this
    

`Datasets` <br/>
 `├──train`  <br/>
     `├──DFWB (synthetic)`   <br/>
          `├──input_crops (clean, Gaussian noised added during training)`   <br/>
     `└──SIDD (real-world)`   <br/>
          `├──input_crops (degraded)`   <br/>

 `└──val`  <br/>
     `├──SIDD`   <br/>
          `├──input_crops`   <br/>
          `└──target_crops`   <br/>

 `└──test`  <br/>
     `├──SIDD`   <br/>
          `├──ValidationNoisyBlocksSrgb.mat`   <br/>
          `└──ValidationGtBlocksSrgb.mat`   <br/>
     `├──DND`   <br/>
          `├──info.mat`   <br/>
          `└──images_srgb`   <br/>
               `├──0001.mat`   <br/>
               `├──0002.mat`   <br/>
               `├── ...    `   <br/>
               `└──0050.mat` 