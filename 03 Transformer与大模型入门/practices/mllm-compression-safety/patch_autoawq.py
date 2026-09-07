"""Apply the two historical LLaVA calibration fixes to AutoAWQ 0.2.7.post3."""
from importlib.metadata import distribution
import ast


def patched(source):
    replacements = [
        ('self.model(samples.to(next(self.model.parameters()).device))',
         "calibration_model = getattr(self.model, 'language_model', self.model)\n            calibration_model(samples.to(next(calibration_model.parameters()).device))"),
        ('self.inps = self.inps.to(common_device)',
         "self.module_kwargs['past_key_value'] = None\n            self.inps = self.inps.to(common_device)"),
    ]
    for old,new in replacements:
        if new in source:
            continue
        if source.count(old) != 1:
            raise ValueError('Unexpected AutoAWQ source; no files changed')
        source=source.replace(old,new)
    ast.parse(source)
    return source


if __name__=='__main__':
    package=distribution('autoawq')
    if package.version != '0.2.7.post3':
        raise SystemExit('Use the isolated AutoAWQ 0.2.7.post3 environment described in README')
    path=package.locate_file('awq/quantize/quantizer.py')
    original=path.read_text(encoding='utf-8')
    updated=patched(original)
    if updated!=original:
        backup=path.with_suffix('.py.before-dl-notes')
        if backup.exists():
            raise FileExistsError(f'Existing backup: {backup}; inspect before retrying')
        backup.write_text(original,encoding='utf-8')
        temp=path.with_suffix('.py.tmp')
        temp.write_text(updated,encoding='utf-8')
        temp.replace(path)
    print('AutoAWQ LLaVA calibration fixes ready:',path)
