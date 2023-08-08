# generate enhanced sample
python cut_mix.py
python adv.py
# generate field with original images
python generate_field.py --exp=1 -b=0 -r=./result_field_featuremap_tk -i=reference_data_1000 -p=result_1000 -g=0
python compare_field.py -d=0 -r=./result_field_featuremap_tk -p=result_1000 -g=0
# generate field with adversarial examples
python generate_field.py --exp=1 -b=0 -r=./result_field_featuremap_tk -i=reference_data_adv0_1000 -p=result_adv0_1000 -g=0
python compare_field.py -d=0 -r=./result_field_featuremap_tk -p=result_adv0_1000 -g=0
# generate field with cutmix images
python generate_field.py --exp=1 -b=0 -r=./result_field_featuremap_tk -i=reference_data_cutmix_1000 -p=result_cutmix_1000 -g=0
python compare_field.py -d=0 -r=./result_field_featuremap_tk -p=result_cutmix_1000 -g=0