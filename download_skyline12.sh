curl https://ttic.uchicago.edu/~smaji/projects/skylineParsing/skyline12.tar.gz --output skyline12.tar.gz
mkdir -p datasets/skyline12
tar xf skyline12.tar.gz -C datasets/skyline12/
rm skyline12.tar.gz