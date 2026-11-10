"""Check iMSE against direct Gaussian scoring and the original research kernels."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from imse.detectors import imse


def original_functions(path, names, scope):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return SimpleNamespace(**scope)


def historical_parity(history, neighborhood, raw):
    g03 = original_functions(history / "g03.py", {"neighborhood", "oas"},
                             dict(torch=torch, F=F, EPS=imse.EPS))
    paired = original_functions(history / "displacement.py", {"paired_generate"},
                                dict(torch=torch, F=F, g03=g03))
    boundary, indices = g03.neighborhood(raw, "cosine", [128, 64, 32, 4], 128, 128)
    for k, tau in ((4, 1.), (16, 1.), (64, 1.), (16, .25)):
        z, parents = paired.paired_generate(
            raw, boundary, indices, dict(rho=1., tau=tau, M=k))
        expected = F.normalize(z, dim=1).double() - F.normalize(raw, dim=1)[parents].double()
        assert torch.equal(expected, imse.class_shifts(raw, k, tau)), (k, tau)
    moment = imse.scatter(raw.double()) / len(raw)
    assert torch.equal(imse._oas(moment, len(raw)), g03.oas(moment, len(raw))[0])
    if neighborhood:
        from imse.common import close_check, require

        refs = original_functions(neighborhood / "geometry.py", {"effective", "class_covariances"},
                                  dict(torch=torch, F=F, core=SimpleNamespace(
                                      require=require, close_check=close_check), method=imse, EPS=imse.EPS))
        for k in ("all", 128, 256, 512):
            _, expected, _ = refs.class_covariances(raw, [k])
            actual = imse.scatter(imse.class_shifts(raw, k, 1.))
            assert torch.equal(actual, expected[str(k)]), k
    print("PASS: original BOSA/LMSE finite-k shifts and OAS, later iMSE neighborhood moments")


def check_detector():
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(21)
    x = torch.randn(450, 32, generator=rng)
    y = torch.arange(len(x)) % 3
    z = torch.randn(15, x.shape[1], generator=rng)
    logits = torch.randn(len(z), 3, generator=rng)
    q = F.normalize(x, dim=1).double()
    means = torch.stack([q[y == c].mean(0) for c in range(3)])
    residual = q - means[y]
    cov = imse._oas(residual.T @ residual / len(x), len(x))
    for k in (16, "all"):
        scatter = sum(imse.scatter(imse.class_shifts(x[y == c], k, 1.)) for c in range(3))
        inflated = cov + scatter / len(x)
        r = F.normalize(z, dim=1).double() - means[logits.argmax(1)]
        id_quad = (r * torch.linalg.solve(cov, r.T).T).sum(1)
        alt_quad = (r * torch.linalg.solve(inflated, r.T).T).sum(1)
        expected = .5 * (torch.linalg.slogdet(inflated).logabsdet
                         - torch.linalg.slogdet(cov).logabsdet - id_quad + alt_quad)
        actual = imse.IMSE(k=k).fit(x, y).score(z, logits)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    readme = Path(__file__).resolve().parents[1] / "README.md"
    code = Path(imse.__file__).read_text(encoding="utf-8").strip()
    assert code in readme.read_text(encoding="utf-8"), "README source snippet is out of sync"
    print("PASS: public iMSE API matches the direct Gaussian ratio at k=16 and k=all")
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, help="Original displacement notebook source directory")
    parser.add_argument("--neighborhood", type=Path, help="Original iMSE neighborhood source directory")
    args = parser.parse_args()
    x = check_detector()
    if args.history:
        historical_parity(args.history, args.neighborhood, x)
