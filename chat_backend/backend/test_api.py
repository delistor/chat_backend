"""Quick API test script"""
import requests, json

base = 'http://localhost:8000'

# Test 1: Algorithms
print("=== Test 1: GET /api/algorithms ===")
r = requests.get(f'{base}/api/algorithms')
print(f"Status: {r.status_code}")
algos = r.json()
print(f"Algorithms: {len(algos)}")
for a in algos:
    print(f"  {a['id']}: {a['name']}")

# Test 2: Run KMeans
print("\n=== Test 2: POST /api/algorithm/run (kmeans) ===")
r = requests.post(f'{base}/api/algorithm/run', data={
    'algorithm': 'kmeans',
    'params': json.dumps({'k': 3})
})
d = r.json()
print(f"Status: {r.status_code}")
print(f"Type: {d.get('type')}")
print(f"Message: {d.get('message','')[:80]}")
print(f"Results: {len(d.get('results',[]))}")
for res in d.get('results', []):
    rtype = res.get('type')
    name = res.get('name', '')
    rid = res.get('id', '')
    if rtype == 'image':
        src = res.get('src', '')
        print(f"  {rid}: {rtype} - {name} -> src={src[:60]}...")
    elif rtype == 'table':
        cols = res.get('columns', [])
        rows = len(res.get('rows', []))
        print(f"  {rid}: {rtype} - {name} ({len(cols)} cols, {rows} rows)")
    elif rtype == 'document':
        content_len = len(res.get('content', ''))
        print(f"  {rid}: {rtype} - {name} ({content_len} chars)")
    else:
        print(f"  {rid}: {rtype} - {name}")

# Test 3: Image serving
print("\n=== Test 3: Image file serving ===")
img_results = [r for r in d.get('results', []) if r.get('type') == 'image']
if img_results:
    src = img_results[0].get('src', '')
    url = base + src if src.startswith('/') else src
    ir = requests.get(url)
    print(f"Image URL: {url[:60]}...")
    print(f"Image Status: {ir.status_code}")
    print(f"Image Content-Type: {ir.headers.get('content-type','')}")
    print(f"Image Size: {len(ir.content)} bytes")

# Test 4: Run PCA
print("\n=== Test 4: POST /api/algorithm/run (pca) ===")
r = requests.post(f'{base}/api/algorithm/run', data={
    'algorithm': 'pca',
    'params': json.dumps({'n_components': 2})
})
d = r.json()
print(f"Type: {d.get('type')}")
print(f"Results: {len(d.get('results',[]))}")
for res in d.get('results', []):
    print(f"  {res.get('id','')}: {res.get('type')} - {res.get('name','')}")

# Test 5: Run data_stats
print("\n=== Test 5: POST /api/algorithm/run (data_stats) ===")
r = requests.post(f'{base}/api/algorithm/run', data={
    'algorithm': 'data_stats',
    'params': json.dumps({})
})
d = r.json()
print(f"Type: {d.get('type')}")
print(f"Results: {len(d.get('results',[]))}")
for res in d.get('results', []):
    print(f"  {res.get('id','')}: {res.get('type')} - {res.get('name','')}")

print("\n=== All tests completed! ===")