# synchronize optimzed results to my load mac
remote_dir="/data16/jiaxin/Interpretation/saved/generated/"
ssh LS16_Remote_2 ls $remote_dir | while read line; do echo $line; done
