# convenience script to train all horizon models
from modeling import train_all
if __name__ == '__main__':
    print('Training models...')
    res = train_all(random_search=False)
    print('Done. Metrics:')
    import json
    print(json.dumps(res, indent=2))
