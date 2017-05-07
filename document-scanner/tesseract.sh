#for the cropped long codes
tesseract --user-patterns ./user_patterns.txt -c tessedit_char_whitelist=123456789ACGHKLMNPRTVWXYZ scannedResized.png stdout --psm 6

# for the cropped short codes
# TODO: add wordlist
tesseract --user-patterns ./user_patterns.txt -c tessedit_char_whitelist=123456789ACGHKLMNPRTVWXYZ scannedResized.png stdout --psm 11