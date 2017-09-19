# Splits all sent_*.txt files into validation, test and train files
# split ratio: train 80%, validation 10%, test 10%

for i in $( ls txt); do
	  num_lines=`wc clean_sentences/sent_$i.txt | sed -r 's/^ +([0-9]*) .*/\1/g'`
	  echo "$i has $num_lines lines"

    num_training=`echo "($num_lines * 0.8)/1" | bc`
	  echo "training sentences: $num_training"

	  num_remaining=$(($num_lines-$num_training))

    num_test=$(($num_remaining/2))
	  echo "test sentences: $num_test"
    echo "validation sentences: $num_test"

	  head -n $num_training clean_sentences/sent_$i.txt > clean_sentences/train_$i.txt
    cat clean_sentences/sent_$i.txt | sed -n "$(($num_training+1)),$(($num_training+$num_test))p" > clean_sentences/val_$i.txt
	  tail -n $num_test clean_sentences/sent_$i.txt > clean_sentences/test_$i.txt

	  echo "----"
done
