(* Read frozen trajectory data only. No solver package or shared definitions. *)
Module[{root = "/Users/k0959535/Dropbox/IntegrabilityProjects/2026 Near N=4/qsccpp/runs/wolfnum_converged_fixtures/exact_seed_g02", trajectory, leading, rs, cs, scale, targets = {1/5}, point, coupling, nc, wp, nd, bits, z, oldScale, newScale, shift, extra, lines, path, records = {}, manifest, sources},
  sources = FileNameJoin[{DirectoryName[root], "sources", #}] & /@ {"trajectory_X2Y.m", "lo_X2Y.m"};
  trajectory = Block[{$ContextPath = {"System`"}, $Context = "WolfNumFixtureData`"}, Get[First[sources]]];
  leading = Block[{$ContextPath = {"System`"}, $Context = "WolfNumFixtureData`"}, Get[Last[sources]]];
  If[!ListQ[trajectory] || !AllTrue[trajectory, AssociationQ] || !AssociationQ[leading], Return[Failure["InvalidSnapshot", <||>]]];
  rs[x_, digits_] := Block[{$MinPrecision = 0, $MaxPrecision = Infinity}, Module[{y = If[Precision[x] === Infinity, N[x, digits], x], k, dg, ex}, k = Min[digits, Max[1, Floor[Precision[y]]]]; If[y == 0, "0", {dg, ex} = RealDigits[Abs[y], 10, k]; If[y < 0, "-", ""] <> "0." <> StringJoin[ToString /@ dg] <> "e" <> ToString[ex]]]];
  cs[x_] := rs[Re[x], nd] <> " " <> rs[Im[x], nd];
  scale[g_, n_] := Append[Join[Flatten[Table[g^(2 Max[0, j - (3/2 - leading["lamL"][[a]])]), {a, 4}, {j, n}]], Flatten[Table[g^(2 Max[0, j - (3/2 + leading["lamU"][[a]])]), {a, 4}, {j, n}]]], 1];
  (* Use a nearby, different coupling at .2 so the comparison exercises Newton. *)
  Do[
    point = First@Select[trajectory, #["g"] === coupling &];
    nc = point["Nc"]; wp = point["prec"]; nd = Ceiling[wp] + 5; bits = Ceiling[wp Log2[10.]] + 8;
    If[!TrueQ[point["Gnorm"] <= 10^-22] || Length[point["z"]] =!= 8 nc + 1 || !VectorQ[point["z"], NumericQ], Return[Failure["InvalidSeed", <|"g" -> coupling|>]]];
    oldScale = scale[point["g"], nc]; newScale = scale[coupling, nc];
    z = point["z"] newScale/oldScale;
    z[[-1]] = (point["Delta"] + 10^-24 - 2)^2;
    shift = 4 + leading["Dm"]; extra = Join[Table[{i, shift + 1, 1}, {i, 4}], {{2, shift + 2, 1}, {4, shift + 2, 1}}];
    path = FileNameJoin[{root, "j3_X2Y_g" <> Switch[coupling, 1/10, "0.1", 1/5, "0.2", 1/2, "0.5"] <> "_nc" <> ToString[nc] <> ".txt"}];
    If[FileExistsQ[path], Return[Failure["ExistingFixture", <|"path" -> path|>]]];
    lines = {"prec " <> ToString[bits], "threads 8", "mode solve", "iters 20", "h " <> rs[10^(-Round[0.4 wp]), 10], "dtol 1e-22", "gtol 1e-22", "g " <> cs[coupling], "dims " <> StringRiffle[ToString /@ {nc, point["NQ"], point["Nsh"], nc + 6}], "lamL " <> StringRiffle[rs[#, 8] & /@ leading["lamL"]], "lamU " <> StringRiffle[rs[#, 8] & /@ leading["lamU"]], "th " <> StringRiffle[cs /@ leading["th"]], "c0 " <> StringRiffle[cs /@ N[leading["AA"], wp]], "d0 " <> StringRiffle[cs /@ {1, 1, 1, 1}], "fix 0", "shift " <> StringRiffle[ToString /@ ConstantArray[shift, 4]], "band " <> ToString[shift], "extra " <> ToString[Length[extra]] <> StringJoin[(" " <> StringRiffle[ToString /@ #]) & /@ extra], "pair 2 1 4 3", "n " <> ToString[Length[z]], "scale " <> StringRiffle[cs /@ N[newScale, wp]], "z " <> StringRiffle[cs /@ z], "D 0", "excl 1 " <> ToString[Length[z] - 1], "drow 1", "modes 1", "tangent 1", "chord 1", "vpar 1", "gauto 1", "mu0 0", "ppmax " <> ToString[nc], "gmask 6 1 1 1 2 2 2 3 3 3 4 4 4"};
    Export[path, StringRiffle[lines, "\n"] <> "\n", "Text"];
    AppendTo[records, <|"input" -> path, "sha256" -> IntegerString[FileHash[path, "SHA256"], 16, 64], "g" -> ToString[coupling, InputForm], "seed_g" -> ToString[point["g"], InputForm], "seed_residual" -> ToString[point["Gnorm"], InputForm], "dims" -> {nc, point["NQ"], point["Nsh"], nc + 6}, "precision_bits" -> bits, "gtol" -> "1e-22", "seed_transform" -> "coefficient scale(target)/scale(source), saved same-coupling seed, Delta perturbed by 1e-24 to exercise Newton; original vpar/gauto, mu0=0, ppmax=Nc, gmask and consres=False; fixed AA unchanged"|>],
    {coupling, targets}];
  manifest = <|"status" -> "PREPARED_NOT_YET_CONVERGENCE_VERIFIED", "state" -> "J3 X2Y", "sources" -> AssociationThread[sources, IntegerString[FileHash[#, "SHA256"], 16, 64] & /@ sources], "exporter_sha256" -> IntegerString[FileHash[FileNameJoin[{root, "preparation.wl"}], "SHA256"], 16, 64], "fixtures" -> records, "timing_accepted" -> False|>;
  path = FileNameJoin[{root, "preparation_metadata.json"}];
  If[FileExistsQ[path], Return[Failure["ExistingManifest", <|"path" -> path|>]]];
  Export[path, manifest, "RawJSON"];
  Grid[Prepend[({#["g"], #["seed_g"], #["dims"], #["precision_bits"], #["gtol"], "prepared; convergence pending"} & /@ records), {"Target g", "Seed g", "Dimensions", "Bits", "gtol", "Status"}], Frame -> All, Alignment -> Left]
]
