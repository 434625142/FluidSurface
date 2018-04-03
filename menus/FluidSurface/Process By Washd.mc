Duplicate>{'name': 'surface', 'stack': True}
Fast DOG>{'sigma': 5.0, 'stack': True}
Watershed Surface>{'stack': True}
Image Calculator>{'img1': 'sequence', 'op': 'max', 'img2': 'surface'}
Kill Image>{'name': 'surface'}
Find Surface>{'num': 10}
Mark Surface>{'num': 10, 'stack': True}