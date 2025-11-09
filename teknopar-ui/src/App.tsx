import React, { useState } from "react";

// ----- TYPES -----
type MachineType = {
  id: string;
  name: string;
};

type Machine = {
  id: string;
  name: string;
  typeId: string;
};

type Station = {
  id: string;
  name: string;
  size: "big" | "small";
};

type Operation = {
  id: string;
  order: number;
  requiredMachineType: string;
  requiresBigStation: boolean;
  feasibleMachines: string[];
  feasibleStations: string[];
  processingTimes: Record<string, number>;
};

type Job = {
  id: string;
  name: string;
  releaseTime: number;
  dueDate: number;
  needsGrinding: boolean;
  grindingTime: number;
  needsPainting: boolean;
  paintingTime: number;
  operations: Operation[];
};

const App: React.FC = () => {
  const newId = () => Math.random().toString(36).substring(2, 9);

  // 1) Genel
  const [bigM, setBigM] = useState<number>(10000);
  const [bigML, setBigML] = useState<number>(10000);

  // 2) Makine tipleri
  const [machineTypes, setMachineTypes] = useState<MachineType[]>([
    { id: "k1", name: "Kaynak Tip 1" },
  ]);

  // 3) Makineler
  const [machines, setMachines] = useState<Machine[]>([
    { id: "m1", name: "Makine 1", typeId: "k1" },
  ]);

  // 4) İstasyonlar
  const [stations, setStations] = useState<Station[]>([
    { id: "s1", name: "İstasyon 1", size: "big" },
    { id: "s2", name: "İstasyon 2", size: "small" },
  ]);

  // 5) İşler
  const [jobs, setJobs] = useState<Job[]>([
    {
      id: "j1",
      name: "İş 1",
      releaseTime: 0,
      dueDate: 100,
      needsGrinding: false,
      grindingTime: 0,
      needsPainting: false,
      paintingTime: 0,
      operations: [
        {
          id: "j1-op1",
          order: 1,
          requiredMachineType: "k1",
          requiresBigStation: false,
          feasibleMachines: ["m1"],
          feasibleStations: ["s1", "s2"],
          processingTimes: { m1: 10 },
        },
      ],
    },
  ]);

  // ---- ACTIONS ----
  const addMachineType = () => {
    setMachineTypes((prev: MachineType[]) => [
      ...prev,
      { id: newId(), name: `Yeni Tip ${prev.length + 1}` },
    ]);
  };

  const addMachine = () => {
    const defaultType = machineTypes[0]?.id ?? "";
    setMachines((prev: Machine[]) => [
      ...prev,
      {
        id: newId(),
        name: `Makine ${prev.length + 1}`,
        typeId: defaultType,
      },
    ]);
  };

  const addStation = () => {
    setStations((prev: Station[]) => [
      ...prev,
      {
        id: newId(),
        name: `İstasyon ${prev.length + 1}`,
        size: "small",
      },
    ]);
  };

  const addJob = () => {
    setJobs((prev: Job[]) => [
      ...prev,
      {
        id: newId(),
        name: `İş ${prev.length + 1}`,
        releaseTime: 0,
        dueDate: 100,
        needsGrinding: false,
        grindingTime: 0,
        needsPainting: false,
        paintingTime: 0,
        operations: [],
      },
    ]);
  };

  const updateJob = (jobId: string, partial: Partial<Job>) => {
    setJobs((prev: Job[]) =>
      prev.map((j) => (j.id === jobId ? { ...j, ...partial } : j))
    );
  };

  const addOperationToJob = (jobId: string) => {
    setJobs((prev: Job[]) =>
      prev.map((j) => {
        if (j.id !== jobId) return j;
        const nextOrder = j.operations.length + 1;
        return {
          ...j,
          operations: [
            ...j.operations,
            {
              id: newId(),
              order: nextOrder,
              requiredMachineType: machineTypes[0]?.id ?? "",
              requiresBigStation: false,
              feasibleMachines: machines[0] ? [machines[0].id] : [],
              feasibleStations: stations.map((s) => s.id),
              processingTimes: machines[0]
                ? { [machines[0].id]: 10 }
                : {},
            },
          ],
        };
      })
    );
  };

  const updateOperation = (
    jobId: string,
    opId: string,
    partial: Partial<Operation>
  ) => {
    setJobs((prev: Job[]) =>
      prev.map((j) => {
        if (j.id !== jobId) return j;
        return {
          ...j,
          operations: j.operations.map((op) =>
            op.id === opId ? { ...op, ...partial } : op
          ),
        };
      })
    );
  };

  const toggleOpMachine = (
    jobId: string,
    opId: string,
    machineId: string
  ) => {
    setJobs((prev: Job[]) =>
      prev.map((j) => {
        if (j.id !== jobId) return j;
        return {
          ...j,
          operations: j.operations.map((op) => {
            if (op.id !== opId) return op;
            const has = op.feasibleMachines.includes(machineId);
            const nextMachines = has
              ? op.feasibleMachines.filter((m) => m !== machineId)
              : [...op.feasibleMachines, machineId];

            const nextPT: Record<string, number> = { ...op.processingTimes };
            if (has) {
              delete nextPT[machineId];
            } else {
              nextPT[machineId] = 10;
            }

            return {
              ...op,
              feasibleMachines: nextMachines,
              processingTimes: nextPT,
            };
          }),
        };
      })
    );
  };

  const toggleOpStation = (
    jobId: string,
    opId: string,
    stationId: string
  ) => {
    setJobs((prev: Job[]) =>
      prev.map((j) => {
        if (j.id !== jobId) return j;
        return {
          ...j,
          operations: j.operations.map((op) => {
            if (op.id !== opId) return op;
            const has = op.feasibleStations.includes(stationId);
            const nextStations = has
              ? op.feasibleStations.filter((s) => s !== stationId)
              : [...op.feasibleStations, stationId];
            return {
              ...op,
              feasibleStations: nextStations,
            };
          }),
        };
      })
    );
  };

  const setOpProcessingTime = (
    jobId: string,
    opId: string,
    machineId: string,
    value: number
  ) => {
    setJobs((prev: Job[]) =>
      prev.map((j) => {
        if (j.id !== jobId) return j;
        return {
          ...j,
          operations: j.operations.map((op) => {
            if (op.id !== opId) return op;
            return {
              ...op,
              processingTimes: {
                ...op.processingTimes,
                [machineId]: value,
              },
            };
          }),
        };
      })
    );
  };

  const buildPayload = () => {
    const bigStations = stations
      .filter((s) => s.size === "big")
      .map((s) => s.id);
    const smallStations = stations
      .filter((s) => s.size === "small")
      .map((s) => s.id);

    return {
      H: bigM,
      H_L: bigML,
      machineTypes,
      machines,
      stations,
      L_B: bigStations,
      L_S: smallStations,
      jobs: jobs.map((j) => ({
        ...j,
        operations: j.operations.map((op) => ({ ...op })),
      })),
    };
  };

  const payload = buildPayload();

  return (
    <div style={{ padding: "1rem", fontFamily: "sans-serif" }}>
      <h1>Teknopar Çizelgeleme Veri Girişi (TS)</h1>

      {/* Genel */}
      <section style={{ border: "1px solid #ddd", padding: "1rem", marginBottom: "1rem" }}>
        <h2>1) Genel Parametreler</h2>
        <label>
          Big-M (H):
          <input
            type="number"
            value={bigM}
            onChange={(e) => setBigM(Number(e.target.value))}
            style={{ marginLeft: "0.5rem" }}
          />
        </label>
        <br />
        <label>
          Big-M (istasyon) (H_L):
          <input
            type="number"
            value={bigML}
            onChange={(e) => setBigML(Number(e.target.value))}
            style={{ marginLeft: "0.5rem" }}
          />
        </label>
      </section>

      {/* Makine tipleri */}
      <section style={{ border: "1px solid #ddd", padding: "1rem", marginBottom: "1rem" }}>
        <h2>2) Makine Tipleri (K)</h2>
        {machineTypes.map((k) => (
          <div key={k.id} style={{ marginBottom: "0.5rem" }}>
            <input
              type="text"
              value={k.name}
              onChange={(e) =>
                setMachineTypes((prev: MachineType[]) =>
                  prev.map((kk) => (kk.id === k.id ? { ...kk, name: e.target.value } : kk))
                )
              }
            />
            <span style={{ marginLeft: "0.5rem", color: "#555" }}>id: {k.id}</span>
          </div>
        ))}
        <button onClick={addMachineType}>Makine Tipi Ekle</button>
      </section>

      {/* Makineler */}
      <section style={{ border: "1px solid #ddd", padding: "1rem", marginBottom: "1rem" }}>
        <h2>3) Makineler (M)</h2>
        {machines.map((m) => (
          <div key={m.id} style={{ display: "flex", gap: "0.5rem", marginBottom: "0.5rem" }}>
            <input
              type="text"
              value={m.name}
              onChange={(e) =>
                setMachines((prev: Machine[]) =>
                  prev.map((mm) => (mm.id === m.id ? { ...mm, name: e.target.value } : mm))
                )
              }
            />
            <select
              value={m.typeId}
              onChange={(e) =>
                setMachines((prev: Machine[]) =>
                  prev.map((mm) => (mm.id === m.id ? { ...mm, typeId: e.target.value } : mm))
                )
              }
            >
              <option value="">--Tip seç--</option>
              {machineTypes.map((k) => (
                <option key={k.id} value={k.id}>
                  {k.name}
                </option>
              ))}
            </select>
            <span style={{ color: "#555" }}>id: {m.id}</span>
          </div>
        ))}
        <button onClick={addMachine}>Makine Ekle</button>
      </section>

      {/* İstasyonlar */}
      <section style={{ border: "1px solid #ddd", padding: "1rem", marginBottom: "1rem" }}>
        <h2>4) İstasyonlar (𝓛)</h2>
        {stations.map((s) => (
          <div key={s.id} style={{ display: "flex", gap: "0.5rem", marginBottom: "0.5rem" }}>
            <input
              type="text"
              value={s.name}
              onChange={(e) =>
                setStations((prev: Station[]) =>
                  prev.map((ss) => (ss.id === s.id ? { ...ss, name: e.target.value } : ss))
                )
              }
            />
            <select
              value={s.size}
              onChange={(e) =>
                setStations((prev: Station[]) =>
                  prev.map((ss) =>
                    ss.id === s.id ? { ...ss, size: e.target.value as "big" | "small" } : ss
                  )
                )
              }
            >
              <option value="big">Büyük</option>
              <option value="small">Küçük</option>
            </select>
            <span style={{ color: "#555" }}>id: {s.id}</span>
          </div>
        ))}
        <button onClick={addStation}>İstasyon Ekle</button>
      </section>

      {/* İşler */}
      <section style={{ border: "1px solid #ddd", padding: "1rem", marginBottom: "1rem" }}>
        <h2>5) İşler ve Operasyonlar</h2>
        {jobs.map((j) => (
          <div key={j.id} style={{ border: "1px solid #eee", padding: "0.75rem", marginBottom: "1rem" }}>
            <h3>
              {j.name} (id: {j.id})
            </h3>
            <label>
              İş adı:
              <input
                type="text"
                value={j.name}
                onChange={(e) => updateJob(j.id, { name: e.target.value })}
                style={{ marginLeft: "0.5rem" }}
              />
            </label>
            <br />
            <label>
              Release time (r_j):
              <input
                type="number"
                value={j.releaseTime}
                onChange={(e) => updateJob(j.id, { releaseTime: Number(e.target.value) })}
                style={{ marginLeft: "0.5rem" }}
              />
            </label>
            <br />
            <label>
              Due date (d_j):
              <input
                type="number"
                value={j.dueDate}
                onChange={(e) => updateJob(j.id, { dueDate: Number(e.target.value) })}
                style={{ marginLeft: "0.5rem" }}
              />
            </label>
            <br />
            <label>
              Grinding var:
              <input
                type="checkbox"
                checked={j.needsGrinding}
                onChange={(e) => updateJob(j.id, { needsGrinding: e.target.checked })}
                style={{ marginLeft: "0.5rem" }}
              />
            </label>
            {j.needsGrinding && (
              <input
                type="number"
                value={j.grindingTime}
                onChange={(e) => updateJob(j.id, { grindingTime: Number(e.target.value) })}
                placeholder="t^{grind}_j"
                style={{ marginLeft: "0.5rem" }}
              />
            )}
            <br />
            <label>
              Painting var:
              <input
                type="checkbox"
                checked={j.needsPainting}
                onChange={(e) => updateJob(j.id, { needsPainting: e.target.checked })}
                style={{ marginLeft: "0.5rem" }}
              />
            </label>
            {j.needsPainting && (
              <input
                type="number"
                value={j.paintingTime}
                onChange={(e) => updateJob(j.id, { paintingTime: Number(e.target.value) })}
                placeholder="t^{paint}_j"
                style={{ marginLeft: "0.5rem" }}
              />
            )}

            <div style={{ marginTop: "0.75rem" }}>
              <h4>Operasyonlar</h4>
              {j.operations.map((op) => (
                <div
                  key={op.id}
                  style={{ border: "1px dashed #ccc", padding: "0.5rem", marginBottom: "0.5rem" }}
                >
                  <strong>Operasyon {op.order}</strong> (id: {op.id})
                  <br />
                  <label>
                    Gerekli makine tipi (τ_i):
                    <select
                      value={op.requiredMachineType}
                      onChange={(e) =>
                        updateOperation(j.id, op.id, { requiredMachineType: e.target.value })
                      }
                      style={{ marginLeft: "0.5rem" }}
                    >
                      <option value="">--seç--</option>
                      {machineTypes.map((k) => (
                        <option key={k.id} value={k.id}>
                          {k.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <br />
                  <label>
                    Büyük istasyon zorunlu mu? (β_i):
                    <input
                      type="checkbox"
                      checked={op.requiresBigStation}
                      onChange={(e) =>
                        updateOperation(j.id, op.id, { requiresBigStation: e.target.checked })
                      }
                      style={{ marginLeft: "0.5rem" }}
                    />
                  </label>
                  <br />
                  <div style={{ marginTop: "0.5rem" }}>
                    <span>Uygun makineler (𝓜_i):</span>
                    <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginTop: "0.25rem" }}>
                      {machines.map((m) => (
                        <label key={m.id} style={{ border: "1px solid #ddd", padding: "0.25rem" }}>
                          <input
                            type="checkbox"
                            checked={op.feasibleMachines.includes(m.id)}
                            onChange={() => toggleOpMachine(j.id, op.id, m.id)}
                          />{" "}
                          {m.name}
                        </label>
                      ))}
                    </div>
                  </div>
                  {op.feasibleMachines.length > 0 && (
                    <div style={{ marginTop: "0.5rem" }}>
                      <span>İşlem süreleri (p_im):</span>
                      {op.feasibleMachines.map((mId) => (
                        <div key={mId}>
                          {machines.find((mm) => mm.id === mId)?.name || mId}:
                          <input
                            type="number"
                            value={op.processingTimes[mId] ?? 0}
                            onChange={(e) =>
                              setOpProcessingTime(j.id, op.id, mId, Number(e.target.value))
                            }
                            style={{ marginLeft: "0.5rem" }}
                          />{" "}
                          dk
                        </div>
                      ))}
                    </div>
                  )}
                  <div style={{ marginTop: "0.5rem" }}>
                    <span>Uygun istasyonlar (𝓛_i):</span>
                    <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginTop: "0.25rem" }}>
                      {stations.map((s) => (
                        <label key={s.id} style={{ border: "1px solid #ddd", padding: "0.25rem" }}>
                          <input
                            type="checkbox"
                            checked={op.feasibleStations.includes(s.id)}
                            onChange={() => toggleOpStation(j.id, op.id, s.id)}
                          />{" "}
                          {s.name} ({s.size === "big" ? "Büyük" : "Küçük"})
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
              <button onClick={() => addOperationToJob(j.id)}>Operasyon Ekle</button>
            </div>
          </div>
        ))}
        <button onClick={addJob}>İş Ekle</button>
      </section>

      {/* JSON */}
      <section style={{ border: "1px solid #ddd", padding: "1rem", marginBottom: "1rem" }}>
        <h2>6) JSON Çıktısı</h2>
        <pre
          style={{
            background: "#f7f7f7",
            padding: "0.5rem",
            maxHeight: "300px",
            overflow: "auto",
            fontSize: "0.7rem",
          }}
        >
          {JSON.stringify(payload, null, 2)}
        </pre>
      </section>
    </div>
  );
};

export default App;
