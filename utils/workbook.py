from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = ROOT / 'configs' / 'traffic_models.xlsx'
DATASETS = ('PEMS03', 'PEMS04', 'PEMS07', 'PEMS08')
MODEL_ROWS = {'HA': 'HA', 'DCRNN': 'DCRNN', 'STGCN': 'STGCN',
              'GraphWaveNet': 'Graph WaveNet', 'AGCRN': 'AGCRN', 'PDFormer': 'PDFormer',
              'STAEformer': 'STAEformer', 'MLGO': 'MLGO', 'ASTPKEFormer': 'ASTPKEFormer',
              'DGSTFormer': 'DGST-Former'}
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def workbook_rows(path=DEFAULT_WORKBOOK):
    path = Path(path)
    with ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rels = {r.attrib['Id']: r.attrib['Target'] for r in
                ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        result = {}
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).find('s:sheets', NS):
            rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target = rels[rid]
            member = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            rows = []
            for row in ET.fromstring(z.read(member)).findall('.//s:sheetData/s:row', NS):
                cells = {}
                for cell in row:
                    coord = cell.attrib['r']; value = cell.find('s:v', NS)
                    if cell.find('s:f', NS) is not None and value is None:
                        raise ValueError(f'Formula has no cached value: {sheet.attrib["name"]}!{coord}')
                    if cell.attrib.get('t') == 'inlineStr':
                        text = ''.join(cell.find('s:is', NS).itertext())
                    elif cell.attrib.get('t') == 's':
                        text = shared[int(value.text)]
                    else:
                        text = value.text if value is not None else ''
                    cells[re.sub(r'\d', '', coord)] = text
                rows.append((int(row.attrib['r']), cells))
            result[sheet.attrib['name']] = rows
    return result


def _number(value, field, integer=False):
    match = re.search(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', str(value))
    if not match:
        raise ValueError(f'Missing numeric {field}: {value!r}')
    number = float(match.group())
    if integer and (number < 1 or not number.is_integer()):
        raise ValueError(f'{field} must be a positive integer')
    return int(number) if integer else number


def build_config(dataset, model='DGSTFormer', workbook=DEFAULT_WORKBOOK, variant=None, lambda_diss=None):
    if dataset not in DATASETS or model not in MODEL_ROWS:
        raise ValueError(f'Unknown dataset/model: {dataset}/{model}')
    path = Path(workbook).expanduser().resolve()
    sheets = workbook_rows(path)
    matches = [(n, r) for n, r in sheets.get(dataset, []) if r.get('A') == MODEL_ROWS[model]]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {MODEL_ROWS[model]} row in {dataset}')
    row_number, row = matches[0]
    defaults = json.loads((ROOT/'configs/model_defaults.json').read_text(encoding='utf-8'))
    mc = deepcopy(defaults['models'][model])
    mc.update(defaults.get('dataset_model_overrides', {}).get(dataset, {}).get(model, {}))
    tc = deepcopy(defaults['training'])
    if model == 'HA':
        tc.update(batch_size=1024, learning_rate=0., epochs=1, patience=20,
                  optimizer='Adam', loss='mae', scheduler='none', enable_early_stopping=False)
        mc['dropout'] = 0.
    else:
        optimizer = row.get('E', '')
        if optimizer not in ('Adam', 'AdamW'):
            raise ValueError(f'Unsupported workbook optimizer: {optimizer}')
        loss_text = row.get('H', '').lower()
        loss = 'mae_jacobian' if 'diss' in loss_text else 'huber' if 'huber' in loss_text else 'mae' if 'mae' in loss_text else None
        if loss is None:
            raise ValueError(f'Unsupported workbook loss: {row.get("H")}')
        lr_text = row.get('D', '')
        schedule = ('multistep' if ('multistep' in lr_text.lower() or '阶梯' in lr_text)
                    else 'plateau' if '衰减' in lr_text else 'none')
        tc.update(batch_size=_number(row['G'], 'batch size', True),
                  learning_rate=_number(lr_text, 'learning rate'),
                  epochs=_number(row['F'], 'epochs', True), optimizer=optimizer,
                  patience=_number(row['J'], 'patience', True), loss=loss, scheduler=schedule,
                  enable_early_stopping=True)
        mc['dropout'] = _number(row['I'], 'dropout')
        if tc['learning_rate'] <= 0 or not 0 <= mc['dropout'] < 1:
            raise ValueError('Learning rate must be positive and dropout in [0, 1)')
    tc['early_stopping'] = tc['enable_early_stopping']
    source = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'sheet': dataset, 'range': f'A{row_number}:J{row_number}',
              'specified_settings': row,
              'reference_parameters': int(float(row['B'])),
              'reference_epoch_seconds': _number(row['C'], 'reference time') if model != 'HA' else None,
              'reference_columns_note': 'parameter counts and epoch times are reference values',
              'unspecified_defaults_file': 'configs/model_defaults.json'}
    config = {'dataset': dataset, 'data': deepcopy(defaults['data']), 'model': mc, 'training': tc,
              'evaluation': deepcopy(defaults['evaluation']), 'workbook': source,
              'experiment': {'kind': 'main', 'label': f'{dataset}_{model}_full', 'smoke': False, 'profile': 'workbook'}}
    if variant:
        if model != 'DGSTFormer' or variant not in ('Full', 'A1', 'A2', 'A3', 'A4'):
            raise ValueError(f'Unsupported ablation: {model}/{variant}')
        if variant == 'A1': mc['graph_constraint'] = False
        elif variant == 'A2': mc['gamma_mode'] = 'none'
        elif variant == 'A3': mc['transformer_context'] = False
        elif variant == 'A4': tc['lambda_diss'] = 0.
        config['experiment'].update(kind='ablation', label=variant)
    if lambda_diss is not None:
        if model != 'DGSTFormer' or lambda_diss < 0:
            raise ValueError('Dissipation sensitivity requires DGSTFormer and a nonnegative coefficient')
        tc['lambda_diss'] = float(lambda_diss)
        config['experiment'].update(kind='sensitivity', label=f'lambda_diss={lambda_diss}')
    return config
