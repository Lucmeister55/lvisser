awk '{for(i=1; i<=NF; i++) if ($i == 0) {print; exit}}' /data/lvisser/coverage/all_samples.depth.txt

awk 'BEGIN{FS="\t"; OFS="\t"} NR>1 {print $2, $3, $4, $5}'  /data/lvisser/cgi_islands_ucsc.bed\
  | bedtools sort -i - >  cpg_islands_ucsc_cleaned.bed
