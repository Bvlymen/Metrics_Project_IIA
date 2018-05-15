import pandas as pd
import numpy as np
def Latex_table_from_pandas(table1, table2 = None, fontsize = 10, fontspace = 8, caption = 'Insert Caption', Notes = 'Insert Notes', Column_Var = 'Insert Column Variable Name', Row_Var = 'Insert Row Variable',table_label = 0,sf=3, Space=True,  Column_Var2 = 'Insert Column Variable Name', Row_Var2 = 'Insert Row Variable'):
    latex_string =''
    if isinstance(table1, pd.DataFrame):
        print('DataFrame \n\n\n\n\n')
        num_table_cols = table1.shape[1] + 1
        table_formatter = ''.join(['|c' for i in range(table1.shape[1] +1 )]) +'|'
    else:
        print('Array')
        num_table_cols = table1.shape[1] 
        table_formatter = ''.join(['|c' for i in range(table1.shape[1])]) +'|'

    beginner = '\\begin{table}[h!] \n\\hspace*{-1.5cm} \n\\begin{threeparttable} \n\\fontsize{'+str(fontsize) +'}{' + str(fontspace) +'}\\selectfont \n\\caption{'+str(caption)+ '} \n\\label{table:'+str(table_label)+'} \n\\begin{tabular}{' + table_formatter +'} \n\\hline \\hline \n&\\multicolumn{'+str(num_table_cols-1)+'}{|c|}{{\\Large ' + str(Column_Var) + '}} \\\\ \n\\hline \n{\\bfseries '+str(Row_Var)+'}'


    latex_string += beginner

    def Add_Row(Row,  Latex_String,sf=sf,num_table_cols=1):
        Latex_String1 = Latex_String
        for idx, j in enumerate(Row):
            if idx == len(Row)-1:
                if isinstance(j,float):
                    Latex_String1 += '' + str(round(j,sf)) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\n\\hline\n'
                elif isinstance(j,int):
                    Latex_String1 += '' + str(round(j,sf)) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\n\\hline\n'
                else:
                    try:
                        Latex_String1 += '' + str(round(float(j),sf)) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\\\\ \n\\hline\n'
                    except:
                        Latex_String1 += '' + str(j) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\n\\hline\n'

            else:
                if isinstance(j,float):
                    Latex_String1 += ''+str(round(j,sf))+ '&'
                elif isinstance(j,int):
                    Latex_String1 += ''+str(round(j,sf))+ '&' 
                else:
                    try:
                        Latex_String1 += ''+str(round(float(j),sf))+ '&'
                    except:
                        Latex_String1 += ''+str(j)+ '&'
        return Latex_String1

    def Add_bold_Row(Row,Latex_String):
        Latex_String1 = Latex_String
        for idx, j in enumerate(Row):
            if idx == len(Row)-1:
                Latex_String1 += '{\\bfseries \\large ' + '\\_'.join(str(j).split('_')) + '} \\\\'
            else:
                Latex_String1 += '{\\bfseries \\large ' +'\\_'.join(str(j).split('_')) + '} &'
        return Latex_String1
    
    if isinstance(table1, pd.DataFrame):
        latex_string += '&'
        latex_string = Add_bold_Row(table1.columns, latex_string)
        if Space:
            latex_string += '\n'+''.join(['&' for i in range(num_table_cols-1)]) +'\\\\ \n\\hline\n'
        else:
            latex_string+='\n\\hline\n'

        row_names = np.array(table1.index.values)
        row_names = np.array(['{\\bfseries '+'\\_'.join(str(name).split('_'))+'}' for name in row_names])
        array = table1.values
        array = np.concatenate([row_names.reshape((-1,1)), array], axis =1)
    for row in array:
        latex_string = Add_Row(row,latex_string)
    
    
    if table2 is not None:
        latex_string += '\\hline \n\\multicolumn{'+str(num_table_cols)+'}{|c|}{ }\\\\ \n\\multicolumn{'+str(num_table_cols)+'}{|c|}{ }\\\\ \n\\hline'
        
        if isinstance(table2, pd.DataFrame):
            latex_string += '&'
            latex_string = Add_bold_Row(table2.columns, latex_string)
            if Space:
                latex_string += ''.join(['&' for i in range(num_table_cols-1)]) +'\\\\ \n\\hline\n'
            else:
                latex_string+='\n\\hline\n'

            row_names = np.array(table2.index.values)
            row_names = np.array(['{\\bfseries '+'\\_'.join(str(name).split('_'))+'}' for name in row_names])
            array = table2.values
            array = np.concatenate([row_names.reshape((-1,1)), array], axis =1)
        for row in array:
            latex_string = Add_Row(row,latex_string)
        # if isinstance(table2, pd.DataFrame):
        #     latex_string += '&'
        #     latex_string = Add_Row(table2.columns, latex_string)
        #     row_names = np.array(['{\\bfseries '+str(name)+'}' for name in row_names])
        #     array = table2.values
        #     array = np.concatenate([row_names.reshape((-1,1)), array], axis =1)
        # for row in array:
        #     latex_string = Add_Row(row,latex_string)

    ender = ' \n\\end{tabular} \n\\hspace*{-1.5cm} \n\\begin{tablenotes} \n\\small \n\\item - \n\\item $*$ \n\\item $**$ \n\\item $\\dagger$ \n\\item $^{\\ddagger}$ \n\\end{tablenotes} \n\\end{threeparttable} \n\\end{table}'

    latex_string += ender

    print(latex_string)
    return latex_string


def Specific_Latex_table_from_pandas(table1, table2 = None, fontsize = 10, fontspace = 8, caption = 'Insert Caption', Notes = 'Insert Notes', Column_Var = 'Insert Column Variable Name', Row_Var = 'Insert Row Variable',table_label = 0,sf=3, Space=True,  Column_Var2 = 'Insert Column Variable Name', Row_Var2 = 'Insert Row Variable'):
    latex_string =''
    if isinstance(table1, pd.DataFrame):
        print('DataFrame \n\n\n\n\n')
        num_table_cols = table1.shape[1] + 1
        table_formatter = ''.join(['|c' for i in range(table1.shape[1] +1 )]) +'|'
    else:
        print('Array')
        num_table_cols = table1.shape[1] 
        table_formatter = ''.join(['|c' for i in range(table1.shape[1])]) +'|'

    beginner = '\\begin{table}[h!] \n\\hspace*{-1.5cm} \n\\begin{threeparttable} \n\\fontsize{'+str(fontsize) +'}{' + str(fontspace) +'}\\selectfont \n\\caption{'+str(caption)+ '} \n\\label{table:'+str(table_label)+'} \n\\begin{tabular}{' + table_formatter +'} \n\\hline \\hline \n&\\multicolumn{'+str(num_table_cols-1)+'}{|c|}{{\\Large ' + str(Column_Var) + '}} \\\\ \n\\hline \n{\\bfseries '+str(Row_Var)+'}'


    latex_string += beginner

    def Add_Row(Row,  Latex_String,sf=sf,num_table_cols=1):
        Latex_String1 = Latex_String
        for idx, j in enumerate(Row):
            if idx == len(Row)-1:
                if isinstance(j,float):
                    Latex_String1 += '' + str(round(j,sf)) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\n\\hline\n'
                elif isinstance(j,int):
                    Latex_String1 += '' + str(round(j,sf)) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\n\\hline\n'
                else:
                    try:
                        Latex_String1 += '' + str(round(float(j),sf)) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\\\\ \n\\hline\n'
                    except:
                        Latex_String1 += '' + str(j) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\n\\hline\n'

            else:
                if isinstance(j,float):
                    Latex_String1 += ''+str(round(j,sf))+ '&'
                elif isinstance(j,int):
                    Latex_String1 += ''+str(round(j,sf))+ '&' 
                else:
                    try:
                        Latex_String1 += ''+str(round(float(j),sf))+ '&'
                    except:
                        Latex_String1 += ''+str(j)+ '&'
        return Latex_String1

    def Add_bold_Row_Values(Row,  Latex_String,sf=sf,num_table_cols=1):
        Latex_String1 = Latex_String
        for idx, j in enumerate(Row):
            if idx == len(Row)-1:
                if isinstance(j,float):
                    Latex_String1 += '{\\bfseries ' + str(round(j,sf)) + '} \\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\n\\hline\n'
                elif isinstance(j,int):
                    Latex_String1 += '{\\bfseries ' + str(round(j,sf)) + '} \\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\n\\hline\n'
                else:
                    try:
                        Latex_String1 += '{\\bfseries ' + str(round(float(j),sf)) + '} \\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\\\\ \n\\hline\n'
                    except:
                        Latex_String1 += '{\\bfseries ' + str(j) + '} \\\\ \n'+ ''.join(['&' for i in range(num_table_cols-1)]) +'\n\\hline\n'

            else:
                if isinstance(j,float):
                    Latex_String1 += '{\\bfseries '+str(round(j,sf))+ '}&'
                elif isinstance(j,int):
                    Latex_String1 += '{\\bfseries '+str(round(j,sf))+ '}&' 
                else:
                    try:
                        Latex_String1 += '{\\bfseries '+str(round(float(j),sf))+ '} &'
                    except:
                        Latex_String1 += '{\\bfseries '+str(j)+'} &'
        return Latex_String1

    

    def Add_bold_Row(Row,Latex_String):
        Latex_String1 = Latex_String
        for idx, j in enumerate(Row):
            if idx == len(Row)-1:
                Latex_String1 += '{\\bfseries ' + '\\_'.join(str(j).split('_')) + '} \\\\'
            else:
                Latex_String1 += '{\\bfseries ' +'\\_'.join(str(j).split('_')) + '} &'
        return Latex_String1
    
    if isinstance(table1, pd.DataFrame):
        latex_string += '&'
        latex_string = Add_bold_Row(table1.columns, latex_string)
        if Space:
            latex_string += '\n'+''.join(['&' for i in range(num_table_cols-1)]) +'\\\\ \n\\hline\n'
        else:
            latex_string+='\n\\hline\n'

        row_names = np.array(table1.index.values)
        row_names = np.array(['\\_'.join(str(name).split('_')) for name in row_names])
        array = table1.values
        array = np.concatenate([row_names.reshape((-1,1)), array], axis =1)
    for row in array:
        latex_string = Add_bold_Row_Values(row,latex_string)
    
    
    if table2 is not None:
        latex_string += '\\hline \n\\multicolumn{'+str(num_table_cols)+'}{|c|}{ }\\\\ \n\\multicolumn{'+str(num_table_cols)+'}{|c|}{ }\\\\ \n\\hline'
        
        if isinstance(table2, pd.DataFrame):
            latex_string += '&'
            latex_string = Add_bold_Row(table2.columns, latex_string)
            if Space:
                latex_string += ''.join(['&' for i in range(num_table_cols-1)]) +'\\\\ \n\\hline\n'
            else:
                latex_string+='\n\\hline\n'

            row_names = np.array(table2.index.values)
            row_names = np.array(['{\\bfseries '+str(name)+'}' for name in row_names])
            array = table2.values
            array = np.concatenate([row_names.reshape((-1,1)), array], axis =1)
        for row in array:
            latex_string = Add_Row(row,latex_string)
        # if isinstance(table2, pd.DataFrame):
        #     latex_string += '&'
        #     latex_string = Add_Row(table2.columns, latex_string)
        #     row_names = np.array(['{\\bfseries '+str(name)+'}' for name in row_names])
        #     array = table2.values
        #     array = np.concatenate([row_names.reshape((-1,1)), array], axis =1)
        # for row in array:
        #     latex_string = Add_Row(row,latex_string)

    ender = ' \n\\end{tabular} \n\\hspace*{-1.5cm} \n\\begin{tablenotes} \n\\small \n\\item - \n\\item $*$ \n\\item $**$ \n\\item $\\dagger$ \n\\item $^{\\ddagger}$ \n\\end{tablenotes} \n\\end{threeparttable} \n\\end{table}'

    latex_string += ender

    print(latex_string)
    return latex_string

def Add_Individual_Row(Row,  Latex_String,sf=3, row_name = 'insert name'):
    num_table_cols = len(Row)
    Latex_String1 = Latex_String + '{\\bfseries '+ row_name +'}&'
    for idx, j in enumerate(Row):
        if idx == len(Row)-1:
            if isinstance(j,float):
                Latex_String1 += '' + str(round(j,sf)) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols)]) +'\\\\ \n\\hline\n'
            elif isinstance(j,int):
                Latex_String1 += '' + str(round(j,sf)) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols)]) +'\\\\ \n\\hline\n'
            else:
                try:
                    Latex_String1 += '' + str(round(float(j),sf)) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols)]) +'\\\\ \n\\hline\n'
                except:
                    Latex_String1 += '' + str(j) + '\\\\ \n'+ ''.join(['&' for i in range(num_table_cols)]) +'\\\\ \n\\hline\n'

        else:
            if isinstance(j,float):
                Latex_String1 += ''+str(round(j,sf))+ '&'
            elif isinstance(j,int):
                Latex_String1 += ''+str(round(j,sf))+ '&' 
            else:
                try:
                    Latex_String1 += ''+str(round(float(j),sf))+ '&'
                except:
                    Latex_String1 += ''+str(j)+ '&'
    return Latex_String1
    