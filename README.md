### What is SIRIUS TS?

A flask web app that leverages time series algorithms to provide people with the ability to build simple baseline forecasts from within a guided user interface.It provides the user with a 3 step interface that guides them towards building a baseline forecast with Facebook Prophet.

Here is a screenshot of the app after the user has built a forecast:<br/>
<img src="https://raw.githubusercontent.com/garethcull/forecastr/master/static/img/app.png" width="1024" />


### How does this app work?

This app generates a forecast in 3 steps:

1. Upload your csv
2. Configure your initial forecast 
3. View Forecast and Tweak settings

This app doesn't store any data about the contents of the uploaded csv within a database. This is a session based product. 

Once the csv has been uploaded to the app, the data is then stored within temporary variables in the client and data is then sent back and forth between to client and server until the forecast is generated. 

At a high level, data flows like this:<br/>
<img src="https://raw.githubusercontent.com/garethcull/forecastr/master/static/img/data-flow.png" width="1024" />

As an example, Let’s say a user is at Step 1. They’ve decided to try the app and click on the CTA “Browse File” and choose a CSV to upload. The app then parses this data and sends it server side to a python script that calculates some basic statistics about the data before sending it back and then forward to visualize on the second tab (ie. Step 2: Review Data + Setup Model).  




