#!/bin/bash
# Educational Cybersecurity measures purposes: sanitized for safe sharing, review, and classroom-style inspection of the code here.

echo "Installing Chrome..."
apt-get update
apt-get install -y wget gnupg unzip xvfb libxi6 libgconf-2-4

wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list
apt-get update
apt-get install -y google-chrome-stable

CHROME_VERSION=$(google-chrome --version | grep -oP '\d+\.\d+\.\d+\.\d+')
wget -O /tmp/chromedriver.zip "https://storage.googleapis.com/chrome-for-testing-public/${CHROME_VERSION}/linux64/chromedriver-linux64.zip"
unzip /tmp/chromedriver.zip -d /usr/bin/
mv /usr/bin/chromedriver-linux64/chromedriver /usr/bin/chromedriver
chmod +x /usr/bin/chromedriver

apt-get clean
rm -rf /var/lib/apt/lists/*

echo "Chrome version: $(google-chrome --version)"
echo "ChromeDriver version: $(chromedriver --version)"

python necaptcha_solver.py
