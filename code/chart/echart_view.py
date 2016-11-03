#from echarts import Echart, Legend, Bar, Axis

#chart = Echart('GDP', 'This is a fake chart')
#chart.use(Bar('China', [2, 3, 4, 5]))
#chart.use(Legend(['GDP']))
#chart.use(Axis('category', 'bottom', data=['Nov', 'Dec', 'Jan', 'Feb']))
#chart.plot()

import os
import json
import logging
import tempfile
import webbrowser

axis = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
y = [0, 1.04, 2.16, 3.27, 4.31, 5.43, 6.54, 7.66, 8.7, 9.74, 10.86, 11.97, 13.09, 14.13, 15.24, 16.36, 17.47, 18.59, 19.7, 20.82, 21.86, 22.97, 24.01, 25.06, 26.1, 27.21, 28.25, 29.29, 30.41, 31.45, 32.57, 33.68, 34.8, 35.91, 37.03, 38.14, 39.11, 40.22, 41.12, 42.23, 43.27, 44.39, 45.5, 46.54, 47.66, 48.77, 49.89, 51.0, 52.12, 53.23, 54.35,
     55.39, 56.51, 57.62, 58.66, 59.7, 60.59, 61.64, 62.75, 63.87, 64.91, 66.02, 67.14, 68.03, 69.07, 70.11, 71.23, 72.12, 73.16, 74.28, 75.32, 76.28, 77.17, 78.22, 79.26, 80.15, 81.04, 82.01, 83.12, 84.09, 85.13, 85.87, 86.84, 87.66, 88.4, 89.52, 90.41, 91.23, 92.19, 93.01, 93.9, 94.57, 95.46, 96.36, 97.03, 97.55, 98.07, 98.59, 98.96, 99.48, 99.93]


def Process(axis,y):
    with open(os.path.join(os.path.dirname(__file__), 'chart.json')) as f:
        chart_json = f.read().strip('\r')
        print chart_json.strip()
        encode_json = json.loads(chart_json)

        encode_json['series'][0]['data'] = y
        encode_json['xAxis'][0]['data'] = axis
        chart_json = json.dumps(encode_json)
        #print encode_json
        plot(chart_json)

def plot(chart_json):
    html = open('chart/plot.html','w')
    print html
    with open(os.path.join(os.path.dirname(__file__), 'plot.j2')) as f:
        template = f.read()
        #print template
        content = template.replace(
            '{{ opt }}', chart_json)
        html.write(content)
    html.close()
    webbrowser.open('file://' + os.path.realpath(html.name))
    html.close()


#with open(os.path.join(os.path.dirname(__file__), 'chart.json')) as f:
#    chart_json = f.read().strip('\r')
#    print chart_json.strip()
#    plot(chart_json)
if __name__ == "__main__":
    Process(axis,y)