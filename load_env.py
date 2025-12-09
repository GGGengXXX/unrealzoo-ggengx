import os
import argparse
import zipfile
import sys
import shutil
import unrealcv
modelscope = {
    'UE4': 'UnrealZoo/UnrealZoo-UE4',
    'UE5': 'UnrealZoo/UnrealZoo-UE5',
}
binary_linux = dict(
    UE4_ExampleScene='UE4_ExampleScene_Linux.zip',
    UE5_ExampleScene='UE5_ExampleScene_Linux.zip',
    UE4_Collection_Preview='Collection_v4_LinuxNoEditor.zip',
    UnrealZoo_UE5_5_Linux_V1_0_5='UnrealZoo_UE5_5_Linux_V1.0.5',  # Full version
    Textures='Textures.zip'
)

binary_win = dict(
    UE4_ExampleScene='UE4_ExampleScene_Win.zip',
    UE5_ExampleScene='UE5_ExampleScene_Win.zip',
    UnrealZoo_UE5_5_Win64_V1_0_4='UnrealZoo_UE5_5_Win64_V1.0.4',  # Full version
    Textures='Textures.zip'
)

binary_mac = dict(
    UE4_ExampleScene='UE4_ExampleScene_Mac.zip',
    UnrealZoo_UE5_5_Mac_V1_0_3='UnrealZoo_UE5_5_Mac_V1.0.3',  # Full version
    Textures='Textures.zip'
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env", nargs='?', default='Textures',
                        help='Select the binary to download')
    parser.add_argument("-cloud", "--cloud", nargs='?', default='modelscope',
                        help='Select the cloud to download the binary, modelscope or aliyun')
    args = parser.parse_args()

    if 'linux' in sys.platform:
        binary_all = binary_linux
    elif 'darwin' in sys.platform:
        binary_all = binary_mac
    elif 'win' in sys.platform:
        binary_all = binary_win

    # Handle full version names with dots (e.g., UnrealZoo_UE5_5_Linux_V1.0.5)
    # Convert to underscore version for dictionary lookup
    env_key = args.env.replace('.', '_')

    if args.env in binary_all:
        target_name = binary_all[args.env]
    elif env_key in binary_all:
        target_name = binary_all[env_key]
    else:
        print(f"{args.env} is not available to your platform")
        print(f"Available options: {', '.join(binary_all.keys())}")
        exit()

    if args.cloud == 'modelscope':
        if 'UE5' in target_name or 'UnrealZoo_UE5' in target_name:
            remote_repo = modelscope['UE5']
        else:
            remote_repo = modelscope['UE4']
        cmd = f"modelscope download --dataset {remote_repo} --include {target_name} --local_dir ."
        try:
            os.system(cmd)
        except:
            print('Please install modelscope first: pip install modelscope')
            exit()
        filename = target_name

    # Handle full version (may not be a zip file)
    if target_name.endswith('.zip'):
        # Extract zip file
    with zipfile.ZipFile(filename, "r") as z:
        z.extractall()  # extract the zip file
    if 'Textures' in filename:
            folder = 'textures'
        else:
            folder = filename[:-4]  # Remove .zip extension
    else:
        # Full version is already a folder, not a zip
        folder = target_name
        if not os.path.exists(folder):
            # Try to find the folder (ModelScope might have created it with a different name)
            possible_folders = [f for f in os.listdir('.') if os.path.isdir(f) and target_name.split('_')[0] in f]
            if possible_folders:
                folder = possible_folders[0]
            else:
                print(f"Error: Could not find extracted folder for {target_name}")
                exit(1)
    
    target = unrealcv.util.get_path2UnrealEnv()
    print(f"Moving {folder} to {target}")
    if os.path.exists(os.path.join(target, folder)):
        print(f"Warning: {folder} already exists in {target}, skipping move")
    else:
    shutil.move(folder, target)
    
    # Remove zip file if it exists
    if os.path.exists(filename):
    os.remove(filename)

