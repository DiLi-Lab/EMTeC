# Input:
# roi-files in aoi/<textid>.ias (nur ein file pro Text, unabhaengig von Leser, weil Fragen irrelevant) # noqa
# Fixations (Fixation report created by DataViewer software) in <fixfile> mit mindestens den Spalten ['RECORDING_SESSION_LABEL','itemid','CURRENT_FIX_INDEX', 'CURRENT_FIX_X', 'CURRENT_FIX_Y', 'CURRENT_FIX_DURATION', 'CURRENR_FIX_INTEREST_AREA_INDEX'] # noqa
# liste readerIds (hier hart kodiert unten)
# liste textIds (hier hart kodiert unten)
# in readerIds und Fixation report reader 0_1 umbennen in 0
####
# Right/Left: Naechste/vorherige Fixation auswaehlen
# Up/Down:    Fixation nach unten/oben verschieben
# q/w:        Fixation nach rechts/links verschieben
# d/u:        Fixation loeschen (roi=-1) / wieder auf x,y->roi setzen
#
# alle fixs werden auf eine roi gemappt (auch diejenigen, die original auf keiner roi sind) oder muessen geloescht werden. geloeschte fixs werden nicht gespeichert und die restlichen gereindext (1 .. N) # noqa
#
#
# Wenn Ausgabedatei vorhanden, wird Trial uebersprungen (also loeschen, falls neu gemacht werde soll oder falls nicht richtig abgeschlossen wurde) # noqa
#
# in input daten in der zeichenspalte darf kein " oder ' sein (sed "s/[\"']/\*/g")
#
# scale=1, bis gefixt (s. todo)
#
# weil fixations geloescht und gereindext werden koennen, muessen in den ergebnissdateien ALLE spalten drin sein! # noqa
from __future__ import annotations
import argparse
import csv
import os
import yaml
import glob
import tkinter as tk  # malen

import numpy as np
import pandas as pd  # csv lesesn


def makebg(roifile):
    with open(roifile, encoding='ISO-8859-1') as csvfile:
        csvreader = pd.read_csv(csvfile, quoting=csv.QUOTE_NONE, delimiter='\t')
        for _, row in csvreader.iterrows():
            bg.create_rectangle(
                (
                    row['x_left']//scale,
                    row['y_top']//scale,
                    row['x_right']//scale,
                    row['y_bottom']//scale,
                ), tags=('roi', 'rectangle'), outline='slate gray', state='disabled',
            )
            bg.create_text(
                (
                    ((row['x_left']+row['x_right'])/2) // scale,
                    ((row['y_top']+row['y_bottom'])/2) // scale,
                ), text=row['word'], tags=('roi', 'text'), fill='slate gray', state='disabled',
            )


# erwartet 1.untere, 1.obere=2.unteregrenze, 2.obere=3.unteregrenze, ...
def fix2row(y, rowlimits):
    if isinstance(y, str):
        y = float(y.replace(',', '.'))
    if rowlimits == []:
        return -1
    if y < rowlimits[0]:
        return -1
    return fix2row(y, rowlimits[1:])+1


# ROIs auslesen+malen (1.untere, 1.obere=2.untere, ...)
def getlimits(roifile):
    rowlimits = set()
    tmplimits = set()
    with open(roifile, encoding='ISO-8859-1') as csvfile:
        csvreader = pd.read_csv(csvfile, quoting=csv.QUOTE_NONE, delimiter='\t')
        i = 0
        for _, row in csvreader.iterrows():
            i = i+1
            rowlimits.add(row['y_top'])
            rowlimits.add(row['y_bottom'])
            tmplimits.add((row['y_bottom'], row['x_left'], row['x_right']))
    rowlimits = sorted(rowlimits)
    collimits = [set() for _ in range(len(rowlimits)-1)]
    for x in tmplimits:
        collimits[rowlimits.index(x[0])-1].add(x[1])
        collimits[rowlimits.index(x[0])-1].add(x[2])
    collimits = [sorted(x) for x in collimits]
    return rowlimits, collimits


# Generator to iterate over Fixationfiles
def readFixs(fixfile):
    global roifile, roi_filename
    Fixations = pd.read_csv(fixfile, header='infer', delimiter='\t')
    Fixations['CURRENT_FIX_X'] = Fixations['fix_mean_x']
    Fixations['CURRENT_FIX_Y'] = Fixations['fix_mean_y']
    Fixations['CURRENT_FIX_DURATION'] = Fixations['event_duration']
    Fixations['CURRENT_FIX_INDEX'] = Fixations['index']
    Fixations['CURRENT_FIX_INTEREST_AREA_INDEX'] = Fixations['word_roi_id'].replace('.', -2).astype(int) + 1  # noqa: E501
    #Fixations['CURRENT_FIX_INTEREST_AREA_INDEX'] = Fixations['word_roi_id'].replace('.', -2).astype(int)  # noqa: E501
    Fixations.index = Fixations['CURRENT_FIX_INDEX']
    roi_dir = os.path.join(subject_dir, 'aoi')
    for roi_filename in os.listdir(roi_dir):
        # disregard .ias and .pickle files
        if not roi_filename.endswith('.csv'):
            continue
        if roi_filename == f'trialid{Fixations.iloc[0].TRIAL_ID}_{Fixations.iloc[0].item_id}_trialindex{Fixations.iloc[0].Trial_Index_}_coordinates.csv':
            roifile = os.path.join(roi_dir, roi_filename)

    yield Fixations.sort_index()
# Focus fixation


def focus(index):
    bg.coords(focusobj, bg.coords(Fixs.loc[index, 'objid']))
    bg.tag_raise(focusobj)
    bg.tag_raise(Fixs.loc[fix_index, 'objid'])
    bg.tag_raise(Fixs.loc[fix_index, 'objid']+1)
    bg.tag_raise(Fixs.loc[fix_index, 'objid']+2)
    bg.tag_raise(Fixs.loc[fix_index, 'objid']+3)
    bg.tag_raise(Fixs.loc[fix_index, 'objid']+4)


def add(index):
    selected.append(index)
    bg.itemconfig(Fixs.loc[index, 'objid'], state='normal')


def roiindex(x, y):
    for i in range(y):
        # -1, weil die collimits[i][0] die erste untere grenze ist...
        x = x+len(collimits[i])-1
    return x+1  # +1, weil cols bei 0 anfaengt und roiindex bei 1

# Tastaturbindings


def keybindings(event):
    global fix_index
#  print(event.keysym)
    if event.keysym == 'Right':
        if fix_index < Fixs.shape[0]:
            if fix_index > 0:
                bg.itemconfig(Fixs.loc[fix_index, 'objid']+2, state='hidden')
                bg.itemconfig(Fixs.loc[fix_index, 'objid']+3, state='hidden')
                bg.itemconfig(Fixs.loc[fix_index, 'objid']+4, state='hidden')
            fix_index = fix_index+1
            bg.itemconfig(Fixs.loc[fix_index, 'objid'], state='normal')
            bg.itemconfig(Fixs.loc[fix_index, 'objid']+1, state='normal')
            bg.itemconfig(Fixs.loc[fix_index, 'objid']+2, state='normal')
            bg.itemconfig(Fixs.loc[fix_index, 'objid']+3, state='normal')
            bg.itemconfig(Fixs.loc[fix_index, 'objid']+4, state='normal')
            focus(fix_index)
    elif event.keysym == 'Left':
        if fix_index > 0:
            bg.itemconfig(Fixs.loc[fix_index, 'objid'], state='disabled')
            bg.itemconfig(Fixs.loc[fix_index, 'objid']+1, state='disabled')
            bg.itemconfig(Fixs.loc[fix_index, 'objid']+2, state='hidden')
            bg.itemconfig(Fixs.loc[fix_index, 'objid']+3, state='hidden')
            bg.itemconfig(Fixs.loc[fix_index, 'objid']+4, state='hidden')
            fix_index = fix_index-1
            if fix_index > 0:
                bg.itemconfig(Fixs.loc[fix_index, 'objid']+2, state='normal')
                bg.itemconfig(Fixs.loc[fix_index, 'objid']+3, state='normal')
                bg.itemconfig(Fixs.loc[fix_index, 'objid']+4, state='normal')
            focus(fix_index)
    elif event.keysym == 'Up':
        row = Fixs.loc[fix_index, 'line']
        col = Fixs.loc[fix_index, 'index_inline']
        fixindex = Fixs.loc[fix_index, 'roi']
        row = max(0, row-1)
        # Falls links vom ersten -> -1
        col = max(
            0, fix2row(
                Fixs.loc[fix_index, 'CURRENT_FIX_X'], collimits[row],
            ),
        )
        fixindex = roiindex(col, row)
        Fixs.at[fix_index, 'roi'] = fixindex
        Fixs.at[fix_index, 'line'] = row
        Fixs.at[fix_index, 'index_inline'] = col
        rowl = rowlimits
        coll = collimits[row]
        roix = (coll[col]+coll[col+1])/2.
        roiy = (rowl[row]+rowl[row+1])/2.
        bg.coords(
            Fixs.loc[fix_index, 'objid']+2,
            (
                roix//scale,
                roiy//scale,
                Fixs.loc[fix_index, 'CURRENT_FIX_X']//scale,
                Fixs.loc[fix_index, 'CURRENT_FIX_Y']//scale,
            ),
        )
        bg.coords(
            Fixs.loc[fix_index, 'objid']+3, (
                (roix-4)//scale,
                (roiy-4)//scale, (roix+4)//scale, (roiy+4)//scale,
            ),
        )
        bg.coords(
            Fixs.loc[fix_index, 'objid']+4, (
                (roix-4)//scale,
                (roiy+4)//scale, (roix+4)//scale, (roiy-4)//scale,
            ),
        )
    elif event.keysym == 'Down':
        row = Fixs.loc[fix_index, 'line']
        col = Fixs.loc[fix_index, 'index_inline']
        fixindex = Fixs.loc[fix_index, 'roi']
        row = min(len(collimits), row+1)  # Maximal letztes -> row+1
        # Falls links vom ersten -> -1
        col = max(
            0, fix2row(
                Fixs.loc[fix_index, 'CURRENT_FIX_X'], collimits[row],
            ),
        )
        fixindex = roiindex(col, row)
        Fixs.at[fix_index, 'roi'] = fixindex
        Fixs.at[fix_index, 'line'] = row
        Fixs.at[fix_index, 'index_inline'] = col
        rowl = rowlimits
        coll = collimits[row]
        roix = (coll[col]+coll[col+1])/2.
        roiy = (rowl[row]+rowl[row+1])/2.
        bg.coords(
            Fixs.loc[fix_index, 'objid']+2,
            (
                roix//scale,
                roiy//scale,
                Fixs.loc[fix_index, 'CURRENT_FIX_X']//scale,
                Fixs.loc[fix_index, 'CURRENT_FIX_Y']//scale,
            ),
        )
        bg.coords(
            Fixs.loc[fix_index, 'objid']+3, (
                (roix-4)//scale,
                (roiy-4)//scale, (roix+4)//scale, (roiy+4)//scale,
            ),
        )
        bg.coords(
            Fixs.loc[fix_index, 'objid']+4, (
                (roix-4)//scale,
                (roiy+4)//scale, (roix+4)//scale, (roiy-4)//scale,
            ),
        )
    elif event.keysym == 'q':
        row = Fixs.loc[fix_index, 'line']
        col = Fixs.loc[fix_index, 'index_inline']
        fixindex = Fixs.loc[fix_index, 'roi']

        col = max(0, col-1)
        fixindex = roiindex(col, row)
        Fixs.at[fix_index, 'roi'] = fixindex
        Fixs.at[fix_index, 'line'] = row
        Fixs.at[fix_index, 'index_inline'] = col

        rowl = rowlimits
        coll = collimits[row]
        roix = (coll[col]+coll[col+1])/2.
        roiy = (rowl[row]+rowl[row+1])/2.
        bg.coords(
            Fixs.loc[fix_index, 'objid']+2,
            (
                roix//scale,
                roiy//scale,
                Fixs.loc[fix_index, 'CURRENT_FIX_X']//scale,
                Fixs.loc[fix_index, 'CURRENT_FIX_Y']//scale,
            ),
        )
        bg.coords(
            Fixs.loc[fix_index, 'objid']+3, (
                (roix-4)//scale,
                (roiy-4)//scale, (roix+4)//scale, (roiy+4)//scale,
            ),
        )
        bg.coords(
            Fixs.loc[fix_index, 'objid']+4, (
                (roix-4)//scale,
                (roiy+4)//scale, (roix+4)//scale, (roiy-4)//scale,
            ),
        )
    elif event.keysym == 'w':
        row = Fixs.loc[fix_index, 'line']
        col = Fixs.loc[fix_index, 'index_inline']
        fixindex = Fixs.loc[fix_index, 'roi']

        col = min(len(collimits[row]), col+1)
        fixindex = roiindex(col, row)
        Fixs.at[fix_index, 'roi'] = fixindex
        Fixs.at[fix_index, 'line'] = row
        Fixs.at[fix_index, 'index_inline'] = col

        rowl = rowlimits
        coll = collimits[row]
        roix = (coll[col]+coll[col+1])/2.
        roiy = (rowl[row]+rowl[row+1])/2.
        bg.coords(
            Fixs.loc[fix_index, 'objid']+2,
            (
                roix//scale,
                roiy//scale,
                Fixs.loc[fix_index, 'CURRENT_FIX_X']//scale,
                Fixs.loc[fix_index, 'CURRENT_FIX_Y']//scale,
            ),
        )
        bg.coords(
            Fixs.loc[fix_index, 'objid']+3, (
                (roix-4)//scale,
                (roiy-4)//scale, (roix+4)//scale, (roiy+4)//scale,
            ),
        )
        bg.coords(
            Fixs.loc[fix_index, 'objid']+4, (
                (roix-4)//scale,
                (roiy+4)//scale, (roix+4)//scale, (roiy-4)//scale,
            ),
        )
    elif event.keysym == 'd':  # delete
        Fixs.at[fix_index, 'roi'] = -1
        bg.itemconfig(Fixs.loc[fix_index, 'objid'], outline='red')
        bg.itemconfig(Fixs.loc[fix_index, 'objid'], disabledoutline='red')
    elif event.keysym == 'u':  # undelete
        x = float(Fixs.loc[fix_index, 'CURRENT_FIX_X'])
        y = float(Fixs.loc[fix_index, 'CURRENT_FIX_Y'])
        row = fix2row(y, rowlimits)
        row = max(row, 0)
        row = min(row, len(collimits)-1)
        col = fix2row(x, collimits[row])
        col = max(col, 0)
        col = min(col, len(collimits[row])-2)
        rowl = rowlimits
        coll = collimits[row]
        # print(col,len(coll))
        roix = (coll[col]+coll[col+1])/2.
        roiy = (rowl[row]+rowl[row+1])/2.

        Fixs.at[fix_index, 'roi'] = roiindex(col, row)
        Fixs.at[fix_index, 'line'] = row
        Fixs.at[fix_index, 'index_inline'] = col
        bg.itemconfig(Fixs.loc[fix_index, 'objid'], outline='white')
        bg.itemconfig(
            Fixs.loc[fix_index, 'objid'],
            disabledoutline='slate gray',
        )


parser = argparse.ArgumentParser()
parser.add_argument(
    '--basepath',
    type=str,
    default='data/subject_level_data/',
    help='path to the different decoding deploy versions that contain the results',
)
parser.add_argument(
    '--run-on-subj',
    type=str,
    help='Specify the subject ID if fix corr should only run on one subject.',
    default=None,
)
args = parser.parse_args()

scale = 1.
roifile = ''

path_to_config = 'preprocessing/config.yaml'
with open(path_to_config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
subjects_to_exclude = config['exclude']['subjects']


subject_dirs = glob.glob(os.path.join(args.basepath, '*'))

# iterate through the different subjects
for subject_dir in subject_dirs:
    subject_id = subject_dir.split('/')[-1]

    # exclude subjects from the exclude list
    if subject_id in subjects_to_exclude:
        continue

    # if specific subject id is given, run fixation correction only on that one subject
    if args.run_on_subj and subject_id != args.run_on_subj:
        continue

    # path to extracted fixations
    datadir_path = os.path.join(subject_dir, 'fixations', 'event_files')
    # path to save new corrected fixations to
    corr_fix_savepath = os.path.join(subject_dir, 'fixations_corrected', 'event_files')

    for file_name in os.listdir(datadir_path):

        # account for other files and dirs (the plots) in that directory
        if not file_name.endswith('.csv'):
            continue

        fixfile = os.path.join(datadir_path, file_name)

        # retrieve trial info
        orig_fix_csv = pd.read_csv(fixfile, delimiter='\t')
        TRIAL_ID = orig_fix_csv['TRIAL_ID'].unique().item()
        item_id = orig_fix_csv['item_id'].unique().item()
        Trial_Index_ = orig_fix_csv['Trial_Index_'].unique().item()
        model = orig_fix_csv['model'].unique().item()
        decoding_strategy = orig_fix_csv['decoding_strategy'].unique().item()

        file_name_corr_fix = f'{subject_id}-{item_id}-fixations_corrected.csv'

        # Fixations nach reader, text, index durchgehen:
        # generator erzeugen, savebutton next() binden
        for Fixs in readFixs(fixfile):
            if os.path.isfile(
                os.path.join(
                    corr_fix_savepath,
                    file_name_corr_fix
                ),
            ):
                continue
            # Ausgabefenster
            window = tk.Tk()
            window.title('Correcting...')
            bg = tk.Canvas(
                window, width=1680//scale,
                height=1050//scale, background='black',
            )
            bg.bind('<KeyPress>', keybindings)
            bg.pack()
            bg.focus_set()

            # ROIs lesen und malen
            rowlimits, collimits = getlimits(roifile)
            makebg(roifile)
            for rowindex, row in enumerate(rowlimits):
                bg.create_line(
                    (
                        (collimits[rowindex-1][0])//scale, row//scale,
                        (collimits[rowindex-1][-1])//scale, row//scale,
                    ), fill='red',
                )

            old_x = 0
            old_y = 0
            fix_index = 0
            focusobj = bg.create_oval(
                0, 0, 0, 0, tags=(
                    'focus'
                ), fill='green', width=1.7,
            )
            objid = []
            roi = []
            line = []
            index_inline = []
            roiend = []
            for fix in Fixs.itertuples():
                try:
                    x = float(fix.CURRENT_FIX_X.replace(',', '.'))
                    y = float(fix.CURRENT_FIX_Y.replace(',', '.'))
                except AttributeError:
                    x = fix.CURRENT_FIX_X
                    y = fix.CURRENT_FIX_Y
                r = (fix.CURRENT_FIX_DURATION/20)
                row = fix2row(y, rowlimits)
                row = max(row, 0)
                row = min(row, len(collimits)-1)
                col = fix2row(x, collimits[row])
                col = max(col, 0)
                col = min(col, len(collimits[row])-2)
                rowl = rowlimits
                coll = collimits[row]
                # print(col,len(coll))
                roix = (coll[col]+coll[col+1])/2.
                roiy = (rowl[row]+rowl[row+1])/2.
                roi.append(roiindex(col, row))
                line.append(row)
                index_inline.append(col)
                objid.append(
                    bg.create_oval(
                        ((x-r)//scale, (y-r)//scale, (x+r)//scale, (y+r)//scale), tags=(
                            'fix', 'line',
                            'a'+str(fix.Index),
                        ), state='disabled', outline='white', disabledoutline='slate gray', width=1.7,
                    ),
                )
                bg.create_line(
                    (old_x//scale, old_y//scale, x//scale, y//scale), tags=(
                        'fix', 'line',
                        'a'+str(fix.Index),
                    ), state='disabled', fill='white', disabledfill='slate gray',
                )
                bg.create_line(
                    (roix//scale, roiy//scale, x//scale, y//scale), tags=(
                        'fix', 'line',
                        'a'+str(fix.Index),
                    ), state='hidden', fill='white', disabledfill='slate gray', width=4,
                )
                bg.create_line(
                    ((roix-4)//scale, (roiy-4)//scale, (roix+4)//scale, (roiy+4)//scale), tags=(
                        'fix',
                        'line', 'a'+str(fix.Index),
                    ), state='hidden', fill='white', disabledfill='slate gray', width=4,
                )
                bg.create_line(
                    ((roix-4)//scale, (roiy+4)//scale, (roix+4)//scale, (roiy-4)//scale), tags=(
                        'fix',
                        'line', 'a'+str(fix.Index),
                    ), state='hidden', fill='white', disabledfill='slate gray', width=4,
                )
                old_x = x
                old_y = y
            Fixs['objid'] = objid
            Fixs['line'] = line
            Fixs['roi'] = roi
            Fixs['index_inline'] = index_inline

            selected = []
    # Einruecken fuer for-loop..
            window.mainloop()
            # print(Fixs)
    #  Fixs['roiend']=[Fixs['roi']-[]Fixs['roi']]
    #         Fixs.to_csv(
    #             path_or_buf='blub.tst', sep='\t', na_rep='.', float_format=None, columns=None, header=True,
    #             index=True, index_label=None, mode='w', encoding=None, compression=None, quoting=None,
    #             quotechar='"', line_terminator='\n', chunksize=None, date_format=None,
    #             doublequote=True, escapechar=None, decimal='.',
    #         )  # TODO: scale bringt roimappings durcheinander, scale=1 bis gefixt
            Fixs['line'] = Fixs['line']+1  # damits konsistent bei 1 losgeht
            Fixs['index_inline'] = Fixs['index_inline']+1
            NFix = Fixs.shape[0]
            # als geloescht markierte (-1) fuer ausgabe loeschen
            Fixs = Fixs[Fixs['roi'] > 0]
            # alten CURRENT_FIX_INDEX speichern um bei Bedarf mit Originaldaten mappen zu kennen
            Fixs['ORIGINAL_CURRENT_FIX_INDEX'] = Fixs.index
            Fixs.index = [i+1 for i in range(Fixs.shape[0])]  # reindex
            os.system(
                'echo "Number of deleted Fixations in "'+
                ': '+str(NFix-Fixs.shape[0])+' >> trimFix.log',
            )

            Fixs['Fix_adjusted'] = Fixs['roi'] != Fixs['CURRENT_FIX_INTEREST_AREA_INDEX']

            # we have added +1 to the origianl fixation locations in the beginning, but in the uncorrected files
            # the fixation locations start with 0 for the first word; and the word ids from the model output
            # also start with 0. so subtract 1 again from the correct rois
            Fixs['roi'] = Fixs['roi'] - 1

            Fixs = Fixs.rename(columns={'word_roi_id': 'original_word_roi_id', 'word_roi_str': 'original_word_roi_str'})
            Fixs = Fixs.rename(columns={'roi': 'word_roi_id'})

            # open file with word to rois
            roi_csv = pd.read_csv(roifile, delimiter='\t')
            # dictionary to map from the new corrected rois to the words
            mapping_dict = roi_csv.set_index('word_index')['word'].to_dict()

            Fixs['word_roi_str'] = Fixs['word_roi_id'].map(mapping_dict)

            os.system(
                'echo "Number of adjusted Fixations in "'+
                str(np.sum(Fixs['Fix_adjusted']))+' >> trimFix.log',
            )
            #Fixs['RECORDING_SESSION_LABEL'] = f"{reader}_{roi_filename.split('.')[0]}"
            #Fixs['itemid'] = roi_filename
            Fixs = Fixs.drop(
                [
                    'CURRENT_FIX_INTEREST_AREA_INDEX',
                    'CURRENT_FIX_X', 'CURRENT_FIX_Y', 'objid',
                ], axis=1,
            )
            if not os.path.exists(corr_fix_savepath):
                os.makedirs(corr_fix_savepath)
            #os.makedirs('corrected-fixations', exist_ok=True)
            #save_path = os.path.join("corrected-fixations", fr"reader{reader}_{roi_filename.split('.')[0]}.justfix")
            save_path = os.path.join(corr_fix_savepath, file_name_corr_fix)
            Fixs.to_csv(
                path_or_buf=save_path,
                sep='\t', na_rep='.', float_format=None, header=True, index=False,
                index_label='CURRENT_FIX_INDEX', mode='w', encoding=None, compression=None, quoting=None,
                quotechar='"', line_terminator='\n', chunksize=None, date_format=None, doublequote=True,
                escapechar=None, decimal='.',
            )
