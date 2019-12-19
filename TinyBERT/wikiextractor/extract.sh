#!/bin/bash
#
# NOTES
#
# - Must expand templates to avoid a large loss of content.
# - Text will not (redundantly) contain the title string.
# - Keep sections. Section title will be marked by "Section::::".
# - Keep lists. List bullets will be marked by "BULLET::::".
# - Keep tables. They're mostly garbage but can be removed later (remove "^!*").
# - Remove disambiguation pages. Right now there is no use for them.

INPUT="/home/data1/ftpdata/DataResources/enwiki-20191201-pages-articles-multistream.xml.bz2"
#PROCESSES=$2
#TEMPLATES=$3
OUTPUT="/home/data1/ftpdata/DataResources/enwiki-20191201-pages-articles-multistream_extract"

python WikiExtractor.py $INPUT \
       --json \
       --output $OUTPUT \
       --bytes 10M \
       --compress \
       --links \
       --sections \
       --lists \
       --keep_tables \
       --min_text_length 0 \
       --filter_disambig_pages
