dir=/data15/jiaxin/myCode/InterpretationFolder/Interpretation0330/saved/generated
dst=/data15/jiaxin/myCode/InterpretationFolder/Interpretation0330/saved/pack
mkdir $dst
for file in `ls $dir`
do
    subdir=$dir/$file
    start=$dir/$file/`ls $subdir | sort -n | sed -n '1p'`
    end=$dir/$file/`ls $subdir | sort -n | sed -n '$p'`
    echo $start
    echo $end

    mkdir $dst/$file
    cp -rv $start  $dst/$file
    cp -rv $end $dst/$file
done
