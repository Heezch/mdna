#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np


def _list_numeric_dry_ids(folder: Path):
    ids = []
    for xtc_path in folder.glob('dry_*.xtc'):
        suffix = xtc_path.stem.replace('dry_', '')
        if suffix.isdigit() and (folder / f'dry_{suffix}.pdb').exists():
            ids.append(int(suffix))
    return sorted(ids)


def _load_joined_traj(folder: Path, ids):
    trajectories = []
    for idx in ids:
        xtc = folder / f'dry_{idx}.xtc'
        pdb = folder / f'dry_{idx}.pdb'
        trajectories.append(md.load(str(xtc), top=str(pdb)).remove_solvent())
    return md.join(trajectories)


def _save_single_pdb(traj, file_path: Path):
    frame = traj[0] if len(traj) > 1 else traj
    frame.save_pdb(str(file_path))


def _resolve_source_for_site(site_name, frame_idx, n_s1):
    if site_name == 's1':
        return 's1s1', frame_idx
    if site_name in {'s2', 's2_end'}:
        return 's2s2', frame_idx
    if frame_idx < n_s1:
        return 's1s1', frame_idx
    return 's2s2', frame_idx - n_s1


def _accumulate_source_frames(requirements, n_s1):
    site_sources = {}
    source_frames = {}
    for site_name, frame_idx in requirements.items():
        source_name, source_frame = _resolve_source_for_site(site_name, frame_idx, n_s1)
        site_sources[site_name] = source_name
        if source_name in source_frames and source_frames[source_name] != source_frame:
            raise ValueError(
                f'Conflicting source frame requirements for {source_name}: '
                f'{source_frames[source_name]} vs {source_frame}.'
            )
        source_frames[source_name] = source_frame
    return site_sources, source_frames


def main():
    parser = argparse.ArgumentParser(
        description='Extract minimal fixed-mode filament dataset as single-frame PDB files.'
    )
    parser.add_argument('--data-root', type=Path, default=Path('examples/data'))
    parser.add_argument('--output-dir', type=Path, default=Path('examples/data/filament_minimal'))
    parser.add_argument('--stride', type=int, default=100, help='Downsampling stride used by SiteMapper.')
    parser.add_argument(
        '--trajectory-ids',
        type=int,
        nargs='*',
        default=None,
        help='Optional numeric dry_* trajectory IDs to include (default: all numeric dry_*.xtc with matching pdb).',
    )
    parser.add_argument('--dna-frame-idx', type=int, default=1, help='Frame index used by Assembler.add_dna default.')
    args = parser.parse_args()

    data_root = args.data_root
    s1s1_dir = data_root / '0_s1s1'
    s2s2_dir = data_root / '1_s2s2'
    fi_dir = data_root / 'FI'
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.trajectory_ids is None or len(args.trajectory_ids) == 0:
        ids_s1 = _list_numeric_dry_ids(s1s1_dir)
        ids_s2 = _list_numeric_dry_ids(s2s2_dir)
        ids_fi = _list_numeric_dry_ids(fi_dir)
        common_ids = sorted(set(ids_s1).intersection(ids_s2).intersection(ids_fi))
        if not common_ids:
            raise ValueError('No common numeric dry_* trajectories found across 0_s1s1, 1_s2s2, and FI.')
        ids = common_ids
    else:
        ids = sorted(set(args.trajectory_ids))

    print(f'Using trajectory IDs: {ids}')
    print('Loading source trajectories...')
    s1s1 = _load_joined_traj(s1s1_dir, ids)
    s2s2 = _load_joined_traj(s2s2_dir, ids)
    dna_complex = _load_joined_traj(fi_dir, ids)

    segments_overlap = 2
    segments = {
        's1': np.arange(0, 41 + segments_overlap),
        'h3': np.arange(41 - segments_overlap, 53 + segments_overlap),
        's2': np.arange(53 - segments_overlap, 82 + segments_overlap),
        'l2': np.arange(82 - segments_overlap, 95 + segments_overlap),
        'dbd': np.arange(95 - segments_overlap, 137),
    }

    from sys import path as sys_path

    sys_path.append(str(Path(__file__).resolve().parents[1] / 'modules'))
    from filament import SiteMapper

    mapper = SiteMapper(s1s1, s2s2, segments=segments, k=args.stride)
    site_map = mapper.get_site_map()

    fixed_start_requirements = {'s2': 40, 'h3': 90}
    fixed_extend_requirements = {'s1': 20, 'h3': 20, 's2': 20, 'l2': 20, 'dbd': 20}

    n_s1 = len(site_map['s1'])
    start_site_sources, start_source_frames = _accumulate_source_frames(fixed_start_requirements, n_s1=n_s1)
    extend_site_sources, extend_source_frames = _accumulate_source_frames(fixed_extend_requirements, n_s1=n_s1)

    if 's1s1' not in start_source_frames:
        start_source_frames['s1s1'] = extend_source_frames.get('s1s1', 0)
    if 's2s2' not in start_source_frames:
        start_source_frames['s2s2'] = extend_source_frames.get('s2s2', 0)
    if 's1s1' not in extend_source_frames:
        extend_source_frames['s1s1'] = start_source_frames.get('s1s1', 0)
    if 's2s2' not in extend_source_frames:
        extend_source_frames['s2s2'] = start_source_frames.get('s2s2', 0)

    print('Extracting minimal source frames (s1s1/s2s2/dna_complex)...')
    _save_single_pdb(mapper.s1s1[start_source_frames['s1s1']], out_dir / 's1s1_start.pdb')
    _save_single_pdb(mapper.s2s2[start_source_frames['s2s2']], out_dir / 's2s2_start.pdb')
    _save_single_pdb(mapper.s1s1[extend_source_frames['s1s1']], out_dir / 's1s1_extend.pdb')
    _save_single_pdb(mapper.s2s2[extend_source_frames['s2s2']], out_dir / 's2s2_extend.pdb')

    if args.dna_frame_idx >= len(dna_complex):
        raise IndexError(
            f'Requested DNA frame {args.dna_frame_idx}, but only {len(dna_complex)} frames available in joined FI data.'
        )
    _save_single_pdb(dna_complex[args.dna_frame_idx], out_dir / f'complex_frame_{args.dna_frame_idx}.pdb')

    manifest = {
        'data_root': str(data_root),
        'trajectory_ids': ids,
        'stride': args.stride,
        'segments_overlap': segments_overlap,
        'fixed_start_requirements': fixed_start_requirements,
        'fixed_extend_requirements': fixed_extend_requirements,
        'start_site_sources': start_site_sources,
        'extend_site_sources': extend_site_sources,
        'start_source_frames': start_source_frames,
        'extend_source_frames': extend_source_frames,
        'dna_frame_index': args.dna_frame_idx,
    }

    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)

    print(f'Minimal dataset written to: {out_dir}')
    print('Use Assembler.load_minimal_site_map(...) and segment="minimal" to use this dataset.')


if __name__ == '__main__':
    main()
