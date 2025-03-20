DOWNLOAD_PATH=$1
mkdir $DOWNLOAD_PATH

for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17                                                                                           wuxiaoyang@SH-IDC1-10-140-24-32
do tmux new-session -d -s pano$i && tmux send-keys -t pano$i "cd $DOWNLOAD_PATH && get https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_$i.zip" Enter
done

for i in 01 02 03 04 05 06 07 08 10 11 12 13 14 15 16 17                                                                                           wuxiaoyang@SH-IDC1-10-140-24-32
do tmux new-session -d -s prsp$i && tmux send-keys -t prsp$i "cd $DOWNLOAD_PATH && wget https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_$i.zip" Enter
done