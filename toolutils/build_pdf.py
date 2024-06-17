from fpdf import FPDF
import json
import math


class PDF(FPDF):
    def titles(self, title, backend):
        self.set_xy(0.0, 0.0)
        self.set_font('Times', 'B', 16)
        self.set_text_color(50, 220, 50)
        self.cell(w=210.0, h=40.0, align='C', txt=title +
                  ' REPORT (' + backend + ')', border=0)

    def draw_names(self, name):
        self.y += 5
        self.cell(w=210.0, h=5.0, align='C', ln=2, txt=name, border=0)
        
    def lines(self):
        self.rect(5.0, 5.0, 200.0, 287.0)

    def icon(self, icon_path):
        self.set_xy(10.0, 10.0)
        self.image(icon_path,  link='', type='', w=38, h=8)
        self.set_xy(157.0, 0.0)
        self.set_font('Times', 'B', 10)
        self.set_text_color(220, 50, 50)
        self.cell(w=60.0, h=25.0, align='C', txt='Stream Computing', border=0)

    def charts(self, chart_path):
        self.image(chart_path,  link='', type='', w=700/4, h=450/4.9)

    def diff_tables(self, data, dataset):
        col_width = 45
        x = self.x
        i = 0
        self.set_font("Times",  'B', size=10)
        line_height = self.font_size * 2
        self.x = x + 50
        self.multi_cell(90 * math.ceil(((len(data)) / 3)), line_height,
                        'Accuracy Results' + ' (' + dataset + ')', border=1, align='C')
        y = self.y
        reset_y = self.y
        self.ln(line_height)
        self.set_font("Times", size=10)
        final_y = None
        for i, (key, val) in enumerate(data.items()):
            if i < 7:
                if (i % 3 == 0):
                    final_y = y
                    y = reset_y
                self.x = x + 90 * (i // 3) + 50
                self.y = y
                self.multi_cell(col_width, line_height,
                                key, border=1, align='C')
                self.x += (45 + 90 * (i // 3)) + 50
                self.y = y
                self.multi_cell(col_width, line_height,
                                str(val) + ("%" if key == "Data Percent" else ""), border=1, align='C')
                y = self.y
                i += 1


    def diff_numeric_tables(self, data, dataset):
        col_width = 35
        self.set_font("Times",  'B', size=10)
        line_height = self.font_size * 2
        self.x += 5
        self.y += 5
        self.multi_cell(180, line_height,
                        'Numeric Results' + ' (' + dataset + ')', border=1, align='C')
        y = self.y
        self.ln(line_height)
        self.set_font("Times", size=10)
        self.y = y
        start_x = self.x + 5
        
        col_names = ["index"] + list(data.keys())
        row_names = sorted(list(data[list(data.keys())[0]].keys()))
        row_num = len(row_names)

        # plot row names
        self.x = start_x
        for i, name in enumerate(col_names):
            self.y = y
            self.cell(col_width, line_height, name, border=1, align='C')

        # plot every row
        for i in range(row_num):
            self.y += line_height
            presum = start_x
            self.x = presum
            for j, name in enumerate(col_names):
                if not j:
                    self.cell(col_width, line_height, row_names[i], border=1, align='C')
                    continue
                val = data[name][row_names[i]]
                if isinstance(val, list):
                    temp_val = ", ".join('%.3f' % t for t in val)
                else:
                    temp_val = '%.3f' % val
                self.cell(col_width, line_height, temp_val[:-1], border=1, align='C')
        self.y += line_height
        self.x = start_x - 5

    def graph_tables(self, data):
        real_data = []
        row_name = []
        row_data = []
        for key, val in data.items():
            row_name.append(key)
            row_data.append(str(val))
        real_data.append(row_name)
        real_data.append(row_data)

        col_width = 45
        self.set_xy(10.00125, 20)
        x = self.x
        self.x += 27
        self.set_font("Times",  'B', size=10)
        line_height = self.font_size * 2.5
        self.multi_cell(135, line_height,
                        'Graph Compilation Results', border=1, align='C')
        y = self.y
        self.ln(line_height)
        self.set_font("Times", size=10)
        for row in real_data:
            self.x = x
            for i, datum in enumerate(row):
                self.y = y
                self.x += (i + 1) * 45 - 18
                self.multi_cell(col_width, line_height,
                                str(datum), border=1, align='C')
            y = self.y
        self.y += 5

    def performance_tables(self, data, thread_num, region):
        real_data = []
        row_name = []
        for i in range(len(data)):
            row_data = []
            for key, val in data[i].items():
                if i == 0:
                    row_name.append(key)
                row_data.append(val)
            real_data.append(row_data)
        real_data.insert(0, row_name)

        col_width = 27
        self.set_xy(10.00125, 50)
        x = self.x
        self.x += 7
        self.set_font("Times",  'B', size=10)
        line_height = self.font_size * 2
        self.multi_cell(175, line_height, f'Performance Results [ {region}, thread_num : ' + str(thread_num) + " ]",
                        border=1, align='C')
        y = self.y

        width_table = {0: 54, 1: 27, 2: 27, 3: 27, 4: 40}

        self.ln(line_height)
        self.set_font("Times", size=10)
        for row in real_data:
            self.x = x
            pre_sum = 0
            for i, datum in enumerate(row):
                self.y = y
                self.x += pre_sum + 7
                pre_sum += width_table[i]
                self.multi_cell(width_table[i], line_height,
                                "%.2f" % datum if isinstance(datum, float) else str(datum), border=1, align='C')

            y = self.y

            self.ln(line_height)

    def footer(self):
        # Go to 1.5 cm from bottom
        self.set_y(-15)
        # Select Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Print centered page number
        self.cell(0, 10, '%s' % self.page_no(), 0, 0, 'C')

    def generate_report(self, path):
        with open(path + 'result.json', 'r') as f:
            report = json.load(f)
        icon_path = 'toolutils/icon.PNG'
        self.add_page()
        self.lines()
        self.icon(icon_path)
        self.graph_tables(report['Graph Compile'])
        if 'Performance' in report:
            self.performance_tables(report['Performance'], report["Thread_num"], report["Region"])
        if 'Accuracy' in report:
            numeric = report['Accuracy'].get('Numeric', None)
            diff_dist = report['Accuracy'].get('Diff Dist', None)

            report['Accuracy'].pop('Numeric') if 'Numeric' in report["Accuracy"] else None
            numeric.pop('image_names') if numeric and 'image_names' in numeric else None
            report['Accuracy'].pop('Diff Dist') if 'Diff Dist' in report["Accuracy"] else None

            self.diff_tables(report['Accuracy'], report['Dataset'])
            
            if numeric:
                temp = {}
                for key, val in numeric.items():
                    temp[key] = val
                    if len(temp) >= 4:
                        self.diff_numeric_tables(temp, report['Dataset'])
                        
                        temp = {}
                if len(temp):
                    self.diff_numeric_tables(temp, report['Dataset'])

                for i, name in enumerate(diff_dist):
                    index = name.split('-')[-1].split('.')[0]
                    if index[:3] == "b@b":
                        index = report['Output_name'].split(',')[int(index[3:])]
                    self.charts(path + name)
                    self.draw_names(index)
        self.titles(report['Model'], report['Backend'])
        self.set_author('Stream Computing')
        self.output(path + report['Model'] + '.pdf', 'F')
        return True


def build_pdf(path):
    pdf = PDF(orientation='P', unit='mm', format='A4')
    return pdf.generate_report(path)