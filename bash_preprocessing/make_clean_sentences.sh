# Pipeline:
# - takes all files from language folders "txt/{language_code}/*"
# - uses sentence splitter which is part of the dataset
# - removes all sentences that start with "<" (meta tags that shouldn't be part of training data) and "(" also should be removed
# - removes all sentences shorter than 6 characters
# - removes text in brackets e.g. "This is sentence (text in brackets)." will become "This is sentence ."
# - removes duplicates
# - shuffles
# - saves to clean_sentences folder, each language in separate file

for i in $(ls txt); do
	echo "parsing $i"
	cat txt/$i/* | ./tools/split-sentences.perl -l $i | grep -v "^[<(]" | grep -v -E "^.{0,6}$" | sed "s/(.*)//g" | sed "s/(.*//g" | sort | uniq | shuf > clean_sentences/sent_$i.txt
done
