# Clone repo and enter directory 
git clone git@github.com:pastudan/monopoly-ticket-reader.git && cd monopoly-ticket-reader

# Set up a virtualenv
virtualenv venv

# Install Dependencies
pip install -r requirements.txt

# Run scan script, which takes an input image and writes a bunch of output images to the output-images directory
python scan.py -i input-images/2018-sample-ticket.JPG

# The script finds all gamepieces (large rectangles), de-skews and de-warps them, outputting files along the way.
# - The squared whole gamepieces are saved as 3-warped-X.png, with X being the number of the gamepiece.
# - The 4 game strips on each game piece will be cropped and saved as individual files as 7-bY-manual-code-cropped-X.png, with X being the number of the game piece, and Y being the number of the game strip.
