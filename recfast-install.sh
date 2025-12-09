git clone https://bitbucket.org/Jacetoto/recfast-.vx.git
cd recfast-.vx
make
cd ../
mkdir recfast-output

# need to change the output path in the recfast ini file
sed -i '' 's|path for output \t\t= ./outputs/|path for output \t\t= recfast-output/|' recfast-.vx/runfiles/parameters.ini