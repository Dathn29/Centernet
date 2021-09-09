# Detect image
```bash
$ pip install -r requirements.txt
$ pip install -U torch==1.4 torchvision==0.5 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
$ cd src/lib/models/networks/DCNv2
$ cd src
$ !python demo.py ctdet --demo img_path  --load_model model_path --save save_dir
```
[Link weight: ctdet_coco_dla_2x.pth](https://drive.google.com/file/d/1BWinZg-JBi0DniU6rw9SVvcWg_S5r2FZ/view?usp=sharing)
