"""
auth.py - Registration and verification logic

Registration (site 1 and 2):
    - Takes 3 captured images
    - Runs preprocessing + inference on each
    - Computes average embedding
    - Stores 3 individual + 1 average = 4 embeddings total

Verification (site 1 -- 1:1):
    - Takes 1 image + username
    - Checks average embedding first
    - If fails, checks each of 3 individuals
    - Any single match = accept

Identification (site 2 -- 1:N):
    - Takes 1 image
    - Checks against all users average embeddings
    - Returns closest match if within threshold
"""

import numpy as np
from inference import get_embedding
from embeddings import save_embeddings, load_embeddings, load_all_embeddings, user_exists

# Distance threshold from deployment_config.json (Run 7 scratch best)
import config
THRESHOLD = config.THRESHOLD


def _distance(a, b):
    return float(np.linalg.norm(a - b))


def register(username, images):
    if len(images) != 2:
        return {'success': False, 'message': 'Expected 2 images, got ' + str(len(images))}
    if user_exists(username):
        return {'success': False, 'message': 'User "' + username + '" already registered'}

    print('[auth] Registering user: ' + username)
    embeddings = []
    for i, img in enumerate(images):
        try:
            emb = get_embedding(img)
            embeddings.append(emb)
            print('  Image {}/3 embedded  norm={:.4f}'.format(i + 1, float(np.linalg.norm(emb))))
        except Exception as e:
            return {'success': False, 'message': 'Embedding failed on image ' + str(i + 1) + ': ' + str(e)}

    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-10)

    try:
        save_embeddings(username, embeddings, avg_embedding)
    except Exception as e:
        return {'success': False, 'message': 'Failed to save embeddings: ' + str(e)}

    print('[auth] Registered "' + username + '" -- 4 embeddings saved')
    return {'success': True, 'message': 'User "' + username + '" registered successfully'}


def verify(username, image):
    if not user_exists(username):
        return {'success': False, 'message': 'User "' + username + '" not found',
                'distance': None, 'matched': None}

    print('[auth] Verifying user: ' + username)
    try:
        query_emb = get_embedding(image)
    except Exception as e:
        return {'success': False, 'message': 'Embedding failed: ' + str(e),
                'distance': None, 'matched': None}

    individual_embs, avg_emb = load_embeddings(username)

    # Step 1 -- check average first
    avg_dist = _distance(query_emb, avg_emb)
    print('  Avg distance  : {:.4f}  threshold: {:.4f}'.format(avg_dist, THRESHOLD))
    if avg_dist < THRESHOLD:
        print('  [ACCEPT] matched on average embedding')
        return {'success': True, 'message': 'Verified as "' + username + '"',
                'distance': avg_dist, 'matched': 'average'}

    # Step 2 -- check each individual
    for i, emb in enumerate(individual_embs):
        dist = _distance(query_emb, emb)
        print('  Individual {} distance: {:.4f}'.format(i + 1, dist))
        if dist < THRESHOLD:
            print('  [ACCEPT] matched on individual ' + str(i + 1))
            return {'success': True, 'message': 'Verified as "' + username + '"',
                    'distance': dist, 'matched': 'individual_' + str(i + 1)}

    print('  [REJECT] no match found')
    return {'success': False, 'message': 'Verification failed -- vein pattern does not match',
            'distance': avg_dist, 'matched': None}


def identify(image):
    print('[auth] Running identification...')
    try:
        query_emb = get_embedding(image)
    except Exception as e:
        return {'success': False, 'username': None,
                'message': 'Embedding failed: ' + str(e), 'distance': None}

    all_users = load_all_embeddings()
    if not all_users:
        return {'success': False, 'username': None,
                'message': 'No users registered', 'distance': None}

    best_user = None
    best_dist = float('inf')
    for uname, avg_emb in all_users.items():
        dist = _distance(query_emb, avg_emb)
        print('  {}: {:.4f}'.format(uname, dist))
        if dist < best_dist:
            best_dist = dist
            best_user = uname

    print('  Best match: {}  distance: {:.4f}  threshold: {:.4f}'.format(
        best_user, best_dist, THRESHOLD))

    if best_dist < THRESHOLD:
        print('  [ACCEPT] identified as "' + best_user + '"')
        return {'success': True, 'username': best_user,
                'message': 'Identified as "' + best_user + '"', 'distance': best_dist}

    print('  [REJECT] no match within threshold')
    return {'success': False, 'username': None,
            'message': 'Could not identify -- no matching user found', 'distance': best_dist}
