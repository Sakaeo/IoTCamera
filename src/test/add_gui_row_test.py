import PySimpleGUI as sg

layout = [[sg.Text('My Window')],
          [sg.Text('Click to add to column 1'), sg.B('+', key='-B1-')],
          [sg.Text('Click to add to column 2'), sg.B('+', key='-B2-')],
          [sg.Column([[sg.T('Static')] for i in range(10)], scrollable=True, key='-COL1-'),
           sg.Column([[sg.T('Static')] for i in range(10)], vertical_scroll_only=True, key='-COL2-')]]

window = sg.Window('Window Title', layout)
i = 0
while True:  # Event Loop
    event, values = window.read()
    print(event, values)
    if event in (None, 'Exit'):
        break
    if event == '-B1-':
        window.extend_layout(window['-COL1-'], [[sg.T('A New Input Line'), sg.I(key=f'-IN-{i}-')]])
        i += 1
    elif event == '-B2-':
        window.extend_layout(window['-COL2-'], [[sg.T('A New Input Line'), sg.I(key=f'-IN-{i}-')]])
        i += 1
window.close()
