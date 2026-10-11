# `solvi.experimental.skills`

Learning in a world described by code (experimental): `solvi.experimental.skills.SkillLearner(env, grammar, goals)` —
System 2 plans on an exact model of the environment, verified plans become System 1 skills, each used alone only while
its outcomes keep a certificate; `as_s2(learner)` for `solvi.Agent`; export into the knowledge store. Since 1.1.1:
footprints with variables — `SkillLearner(..., lift="lattice", vocabulary=voc, verify_n=16)` widens
each new skill's footprint (constants to variables, relations widened, atoms dropped) and keeps a wider candidate only
when the plan, instantiated on verification states that match it but not the original, reaches its goal on the exact
copy (`Footprint`, `Var`, `Vocabulary`, `Lift`, `lattice`, `generalize`, `replay`).

::: solvi.experimental.skills
