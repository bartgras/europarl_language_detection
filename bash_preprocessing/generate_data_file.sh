dir='clean_sentences'

generate_dataset(){
    echo "Parsing $1 data"
    for i in $(ls $dir/$1_*.txt); do
        lang_code=`echo $i | sed -r "s/.*${1}_(.+)\.txt/\1/"`
        echo "Parsing: $lang_code"
        cat $i | sed -e "s/^/${lang_code}||/" >> $dir/all_$1_data.txt
    done

    echo "Shuffling $1 data ..."
    cat $dir/all_$1_data.txt | shuf > $dir/all_$1_data_shuffled.txt
    rm $dir/all_$1_data.txt 
}

generate_dataset 'train'
generate_dataset 'test'
generate_dataset 'val'

# generate small copies of the data for debug runs
head -n 500 $dir/all_train_data_shuffled.txt > $dir/all_train_data_shuffled_debug.txt
head -n 500 $dir/all_test_data_shuffled.txt > $dir/all_test_data_shuffled_debug.txt
head -n 500 $dir/all_val_data_shuffled.txt > $dir/all_val_data_shuffled_debug.txt
