"""Child process for "Build your own bot": runs user bot code in N seeded maze games and prints one JSON line.

usage: python -I _bot_runner.py <code_file> <sys_path_json> <n_games> <safety 0|1>"""
import json
import resource
import sys


def set_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 6))
    resource.setrlimit(resource.RLIMIT_AS, (1536 * 2**20, 1536 * 2**20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2**20, 2**20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))     # no new processes or threads from user code


def main():
    set_limits()
    code_file, paths, n, safety = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
    sys.path[:0] = paths
    from games import maze

    with open(code_file) as fh:
        code = fh.read()
    greedy = maze.load_bot(code)
    _, system = maze.build_system(safety, greedy)
    games, abstains = [], 0
    for seed in range(n):
        g = maze.new_game(seed)
        while not g.over:
            mv, resp, note = maze.agent_decide(system, g, safety)
            if note == "abstain":
                abstains += 1
                if abstains == 1:
                    errs = [r.error for r in resp.trace.records if r.name == "greedy_move" and r.error]
                    first_error = errs[0] if errs else resp["move"].why
            maze.step(g, mv, note)
        games.append({"won": g.won, "score": g.score, "dots": maze.TOTAL_DOTS - len(g.dots), "deaths": g.deaths,
                      "ticks": g.tick, "forced": g.forced})
    out = {"summary": maze.summarize(games), "games": games, "abstains": abstains}
    if abstains:
        out["first_error"] = first_error
    print(json.dumps(out))


if __name__ == "__main__":
    main()
