"""Single-owner fixed-cohort staged routing; descriptive checkpoints, no confidence certificate."""
import hashlib
import json
import math
from ...core.runtime import vhash
from ...core.store import JSONLStorage, SQLiteStorage


class StagedTraffic:
    """Return a version for owner execution; never execute/deploy candidate code.

    Predeclare phases: {share, units:[stable IDs], max_excess_loss}. Each unit
    must receive a [0,1] independent actual outcome. Missing labels HOLD.
    A single coordinator owns calls; restart reads the durable change journal.
    """
    def __init__(self, pipeline, version, *, rollout_id, phases):
        self.pipeline, self.journal = pipeline, pipeline.journal
        if type(version) is not int or version < 1 or not isinstance(rollout_id, str) or not rollout_id:
            raise ValueError('promoted version and stable rollout ID required')
        promoted = pipeline._state()[1]
        if version not in promoted or not any(v['version']==version and v['in_force'] for v in pipeline.versions()):
            raise ValueError('candidate must be an active promoted version after fresh gates')
        self.change_id = promoted[version]['change']
        self.version, self.baseline, self.id = version, promoted[version]['parent'], rollout_id
        self.phases = json.loads(json.dumps(phases, allow_nan=False))
        all_units = set()
        if not self.phases:
            raise ValueError('finite phases required')
        for p in self.phases:
            if set(p) != {'share','units','max_excess_loss'} or isinstance(p['share'],bool) or not 0 < p['share'] < 1:
                raise ValueError('phase needs share in (0,1), units and max_excess_loss')
            if isinstance(p['max_excess_loss'],bool) or not isinstance(p['max_excess_loss'],(int,float)) or not math.isfinite(p['max_excess_loss']) or p['max_excess_loss'] < 0:
                raise ValueError('finite nonnegative excess-loss threshold required')
            if not isinstance(p['units'],list) or not p['units'] or any(not isinstance(u,str) or not u or u in all_units for u in p['units']) or len(set(p['units'])) != len(p['units']):
                raise ValueError('predeclared unique stable unit IDs required')
            all_units.update(p['units'])
            arms = [self._arm(u,p['share']) for u in p['units']]
            if not any(arms) or all(arms):
                raise ValueError('each fixed cohort needs both assigned variants')
        self._durable()
        config = dict(version=version, baseline=self.baseline, phases=self.phases)
        self._config_fp = vhash(config)
        old = self._records('rollout_plan')
        if old and any(r['config'] != config for r in old):
            raise ValueError('rollout configuration frozen')
        if not old:self._write('rollout_plan',config=config, fingerprint=vhash(config))

    def _durable(self):
        storage = self.journal.storage
        if isinstance(storage,JSONLStorage):
            if not storage.fsync or storage.rotates or storage._in_batch():
                raise ValueError('nonrotating fsync journal outside a batch required')
        elif isinstance(storage,SQLiteStorage):
            main = next((r[2] for r in storage.db.execute('PRAGMA database_list') if r[1]=='main'),'')
            if not main or storage._batch is not None or storage.db.execute('PRAGMA synchronous').fetchone()[0] < 2 or storage.db.execute('PRAGMA journal_mode').fetchone()[0].lower() in ('off','memory'):
                raise ValueError('persistent synchronous SQLite outside batch required')
        else:
            raise ValueError('durable native journal required')
    def _arm(self,unit,share):
        n=int(hashlib.sha256((self.id+'\0'+unit).encode()).hexdigest(),16)
        return n / 2**256 < share

    def _records(self,action):
        return [r for r in self.journal.records(action) if r.get('rollout')==self.id]

    def _write(self,action,**data):
        return self.journal.write(action,rollout=self.id,**data)

    def _phase(self):
        self._durable()
        if vhash(dict(version=self.version,baseline=self.baseline,phases=self.phases)) != self._config_fp:
            raise ValueError('rollout configuration mutated')
        if self.pipeline._watch_unavailable(self.version) is not None:
            raise ValueError('candidate version no longer active')
        decisions=self._records('rollout_decision')
        if any(r['decision']=='rollback' for r in decisions):return None
        return sum(r['decision']=='advance' for r in decisions)

    def route(self,unit):
        phase=self._phase()
        if phase is None or phase >= len(self.phases):
            raise ValueError('rollout stopped/completed; owner must explicitly choose next deployment')
        p=self.phases[phase]
        if unit not in p['units']:raise ValueError('unit outside frozen current cohort')
        old=[r for r in self._records('rollout_assignment') if r['phase']==phase and r['unit']==unit]
        if old:return old[0]['version']
        version=self.version if self._arm(unit,p['share']) else self.baseline
        self._write('rollout_assignment',phase=phase,unit=unit,version=version)
        return version  # durable BEFORE the owner executes anything

    def observe(self,unit,loss,*,by):
        phase=self._phase()
        if isinstance(loss,bool) or not isinstance(loss,(int,float)) or not math.isfinite(loss) or not 0<=loss<=1:
            raise ValueError('actual bounded loss required')
        assignments=[r for r in self._records('rollout_assignment') if r['phase']==phase and r['unit']==unit]
        if not assignments:raise ValueError('durable assignment required before observation')
        if any(r['unit'] == unit for r in self._observations(phase)):
            raise ValueError('unit already labelled')
        return self.pipeline.observe(self.change_id,dict(rollout=self.id,phase=phase,unit=unit,
                                     assignment=assignments[0]['id'],version=assignments[0]['version'],loss=float(loss)),by=by)

    def _observations(self, phase):
        """Revalidate raw Pipeline.observe records before they can count as evidence."""
        assignments={r['id']:r for r in self._records('rollout_assignment') if r['phase']==phase}
        p=self.phases[phase]
        labels={}
        for record in self.journal.records('observed'):
            effect=record.get('observed')
            if not isinstance(effect,dict) or effect.get('rollout')!=self.id:
                continue
            if type(effect.get('phase')) is not int or effect['phase'] < 0 or effect['phase'] >= len(self.phases):
                raise ValueError('invalid rollout observation phase')
            if effect['phase']!=phase:
                continue
            unit=effect.get('unit')
            assignment=assignments.get(effect.get('assignment'))
            loss=effect.get('loss')
            if (record.get('change')!=self.change_id or not isinstance(unit,str) or unit not in p['units']
                    or assignment is None or assignment['unit']!=unit or assignment['seq']>=record['seq']
                    or type(effect.get('version')) is not int or effect['version']!=assignment['version']
                    or effect['version']!=(self.version if self._arm(unit,p['share']) else self.baseline)
                    or isinstance(loss,bool) or not isinstance(loss,(int,float)) or not math.isfinite(loss) or not 0<=loss<=1):
                raise ValueError('outcome must bind the prior assignment, change, unit, variant and bounded actual loss')
            if unit in labels:
                raise ValueError('duplicate rollout unit outcome')
            labels[unit]=effect
        return list(labels.values())

    def checkpoint(self):
        phase=self._phase()
        if phase is None or phase>=len(self.phases):raise ValueError('rollout stopped/completed')
        p=self.phases[phase]
        labels={r['unit']:r for r in self._observations(phase)}
        missing=[u for u in p['units'] if u not in labels]
        if missing:return dict(decision='hold',phase=phase,missing=missing)
        candidate=[r['loss'] for r in labels.values() if r['version']==self.version]
        baseline=[r['loss'] for r in labels.values() if r['version']==self.baseline]
        diff=sum(candidate)/len(candidate)-sum(baseline)/len(baseline)
        decision='rollback' if diff>p['max_excess_loss'] else 'advance'
        self._write('rollout_decision',phase=phase,decision=decision,excess_loss=diff,labels=len(labels))
        return dict(decision=decision,phase=phase,excess_loss=diff,labels=len(labels))
