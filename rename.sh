for file in /data/lvisser/modkit/outputs/table/data_OHMX20230016R_MM_2/unfiltered/*; do
    mv "$file" "${file/JJN3_2.1/JJN3_2_1}"
done