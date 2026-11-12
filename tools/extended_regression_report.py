#!/usr/bin/env python3
"""Report whether the extended matrix ran, failed, or was queue-deferred."""
import json
import os
import sys
from pathlib import Path


def classify(selection_result, selection, matrix, regression_result):
    """Return a fail-closed summary of selection and aggregate matrix outcomes."""
    try:
        names = json.loads(matrix)
        valid = (isinstance(names, list) and all(isinstance(x, str) and x for x in names)
                 and len(names) == len(set(names)))
    except (TypeError, ValueError):
        names, valid = [], False
    if selection_result != 'success' or not valid:
        return {'outcome': 'failed', 'selected_count': 0, 'ok': False,
                'summary': 'Coverage selection failed or produced an invalid matrix.'}
    if selection == 'deferred' and not names and regression_result == 'skipped':
        return {'outcome': 'deferred', 'selected_count': 0, 'ok': True,
                'summary': 'Deferred: Actions queue busy. No backends were tested.'}
    if selection == 'selected' and names and regression_result == 'success':
        return {'outcome': 'passed', 'selected_count': len(names), 'ok': True,
                'backends': names, 'summary': f"All {len(names)} selected backend jobs passed: {', '.join(names)}."}
    return {'outcome': 'failed', 'selected_count': len(names), 'ok': False,
            'summary': f'Coverage did not complete successfully (selection={selection}, regression={regression_result}).'}


def main():
    result = classify(os.environ.get('SELECTION_RESULT', ''), os.environ.get('COVERAGE_SELECTION', ''),
                      os.environ.get('COVERAGE_MATRIX', ''), os.environ.get('REGRESSION_RESULT', ''))
    print(json.dumps(result, sort_keys=True))
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a') as summary:
            summary.write('## Extended regression coverage\n\n' + result['summary'] + '\n')
    if os.environ.get('GITHUB_OUTPUT'):
        with Path(os.environ['GITHUB_OUTPUT']).open('a') as output:
            output.write(f"outcome={result['outcome']}\nselected_count={result['selected_count']}\n")
    if result['outcome'] == 'deferred':
        print('::notice::' + result['summary'])
    elif not result['ok']:
        print('::error::' + result['summary'])
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
