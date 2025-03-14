import argparse
import glob
import json
import os
import os.path as osp
import sys

import imgviz
import numpy as np
import PIL.Image

import labelme


def _makedir(pa):
    if not osp.exists(pa):
        os.makedirs(pa)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-input_dir', default=r"D:\data\huaxiSkin\0706opt", help='input annotated directory')
    parser.add_argument('-output_dir', default=r"D:\data\huaxiSkin\0706opt_seg", help='output dataset directory')
    parser.add_argument('--labels', default='./labels.txt', help='labels file')
    parser.add_argument(
        '--noviz', help='no visualization', action='store_true'
    )
    args = parser.parse_args()

    _makedir(args.output_dir)
    _makedir(osp.join(args.output_dir, 'JPEGImages'))
    _makedir(osp.join(args.output_dir, 'SegmentationClass'))
    _makedir(osp.join(args.output_dir, 'SegmentationPIL'))
    _makedir(osp.join(args.output_dir, 'SegmentationClassPNG'))

    if not args.noviz:
        _makedir(
            osp.join(args.output_dir, 'SegmentationClassVisualization')
        )
    print('Creating dataset:', args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(args.output_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)
    # print("with input dir:", osp.join(args.input_dir, "*.json"))
    files = os.listdir(args.input_dir)
    for label_file in files:
        if not label_file.endswith(".json"):
            continue
        label_file = osp.join(args.input_dir, label_file)
        print('Generating dataset from:', label_file)
        with open(label_file, encoding="UTF-8") as f:
            base = osp.splitext(osp.basename(label_file))[0]
            out_img_file = osp.join(
                args.output_dir, 'JPEGImages', base + '.jpg')
            out_lbl_file = osp.join(
                args.output_dir, 'SegmentationClass', base + '.npy')
            out_png_file = osp.join(
                args.output_dir, 'SegmentationClassPNG', base + '.png')
            if not args.noviz:
                out_viz_file = osp.join(
                    args.output_dir,
                    'SegmentationClassVisualization',
                    base + '.jpg',
                )

            data = json.load(f)

            img_file = osp.join(osp.dirname(label_file), data['imagePath'])
            img = np.asarray(PIL.Image.open(img_file))
            PIL.Image.fromarray(img).save(out_img_file)
            print(img.shape)
            lbl, ins = labelme.utils.shapes_to_label(
                img_shape=img.shape,
                shapes=data['shapes'],
                label_name_to_value=class_name_to_id,
            )
            labelme.utils.lblsave(out_png_file, lbl)
            opt = np.array(lbl * 120, dtype=np.uint8)
            PIL.Image.fromarray(opt).save(osp.join(args.output_dir, 'SegmentationPIL', base + '.bmp'))

            np.save(out_lbl_file, lbl)

            if not args.noviz:
                viz = imgviz.label2rgb(
                    label=lbl,
                    img=imgviz.rgb2gray(img),
                    font_size=15,
                    label_names=class_names,
                    loc='rb',
                )
                imgviz.io.imsave(out_viz_file, viz)


if __name__ == '__main__':
    main()
