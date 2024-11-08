export CUDA_VISIBLE_DEVICES=0

epochs=50
batch_size=4

class_names=("small_battery",
            "screws_toys",
            "furniture",
            "tripod_plugs",
            "water_cups",
            "keys",
            "pens",
            "locks",
            "screwdrivers",
            "charging_cords",
            "pencil_cords",
            "water_cans",
            "pills",
            "locks",
            "medicine_pack",
            "small_bottles",
            "metal_plate",
            "usb_connector_board")

for class_name in "${class_names[@]}"
    do
        python training.py --class_name $class_name --epochs_no $epochs --batch_size $batch_size 
    done