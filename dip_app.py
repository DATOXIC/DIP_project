"""
DIP — cửa sổ ứng dụng, MỘT màn hình duy nhất:

    bên trái : danh sách người ký (Đạt, Mai, Phi, ...) + tình trạng dữ liệu + "+ Người mới"
    bên phải : 3 nút lớn cho người đang chọn:  ▶ Nhận diện   ● Thu dữ liệu   ✓ Kiểm tra dữ liệu
    phía dưới: nhật ký chữ (kết quả chạy hiện ở đây) + nút Dừng
    thanh menu "Nâng cao": đánh giá, miễn từ, kiểm thử, chẩn đoán camera, hướng dẫn cho người ký

Mở bằng:  py dip.py        (menu chữ cũ:  py dip.py --menu)

Cửa sổ này chỉ là bảng điều khiển: mỗi việc chạy như một chương trình con (cùng các chương trình cũ). Cửa sổ camera của nhận
diện/thu vẫn là cửa sổ OpenCV riêng (phím trong cửa sổ đó như cũ). Mỗi lúc chỉ chạy một việc (camera không dùng chung được).
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog, ttk
from typing import Any, Callable, Dict, List, Optional

import dip
import people

EST = {  # ước lượng thời gian hiển thị trong menu Nâng cao
    "loso": "nhanh", "open": "~15 giây", "baseline": "~13 phút", "hoc": "~23 phút", "hybrid": "~18 phút",
    "enroll": "~10 phút", "confirm": "~4 phút", "adapt": "~11 phút", "citizen": "~1 phút",
}
BIG = ("Segoe UI", 13, "bold")
SMALL = ("Segoe UI", 9)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("DIP — Nhận diện thủ ngữ")
        root.geometry("940x680")
        root.minsize(780, 520)
        self.proc: Optional[subprocess.Popen] = None
        self.q: "queue.Queue[Any]" = queue.Queue()   # (loại, giá trị): LINE | DONE | SCANNED
        self.on_done: Optional[Callable[[int], None]] = None
        self.selected: str = people.OWNER_ID
        self._scanning = False
        self._rescan = False
        self._after: Optional[tuple] = None
        self._build_menu()
        self._build()
        self.refresh_people()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(60, self._poll)

    # ------------------------------------------------------------------ giao diện
    def _build_menu(self) -> None:
        bar = tk.Menu(self.root)
        adv = tk.Menu(bar, tearoff=0)
        ev = tk.Menu(adv, tearoff=0)
        for k, desc in dip.EVAL_KINDS.items():
            ev.add_command(label=f"{desc}  [{EST.get(k, '')}]", command=lambda kk=k: self.evaluate(kk))
        adv.add_cascade(label="Đánh giá (dùng người đang chọn làm người lạ)", menu=ev)
        adv.add_command(label="Miễn từ cho người đang chọn…", command=self.skip_dialog)
        adv.add_command(label="Hướng dẫn cho người ký", command=lambda: messagebox.showinfo("Hướng dẫn cho người ký", dip.HUONG_DAN_NGUOI_KY.strip(), parent=self.root))
        adv.add_separator()
        adv.add_command(label="Quay mẫu âm (video KHÔNG ký, việc thường ngày)…", command=self.neg_record)
        adv.add_command(label="Huấn luyện lại spotter (ghi file mới)…", command=self.neg_train)
        adv.add_command(label="Chấm báo nhầm trên phiên mẫu âm mới nhất", command=self.neg_eval)
        adv.add_separator()
        adv.add_command(label="Chẩn đoán camera", command=lambda: self.run(dip.cmd_camera(), "Chẩn đoán camera"))
        adv.add_command(label="Chạy toàn bộ kiểm thử  [~6 phút]", command=lambda: self.run(dip.cmd_test(), "Kiểm thử"))
        bar.add_cascade(label="Nâng cao", menu=adv)
        self.root.config(menu=bar)

    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)

        # ---- trái: người ký
        left = ttk.LabelFrame(main, text="Người ký", padding=8)
        left.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 10))
        self.tree = ttk.Treeview(left, columns=("tt",), show="tree", height=9, selectmode="browse")
        self.tree.column("#0", width=210)
        self.tree.pack(fill="y", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        ttk.Button(left, text="+ Người mới", command=self.new_person).pack(fill="x", pady=(8, 2))
        self.progress = ttk.Label(left, text="", font=SMALL, foreground="#555", wraplength=210, justify="left")
        self.progress.pack(anchor="w", pady=(6, 0))

        # ---- phải: người đang chọn + 3 nút lớn
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="new")
        self.title_l = ttk.Label(right, text="", font=("Segoe UI", 16, "bold"))
        self.title_l.pack(anchor="w")
        self.info = ttk.Label(right, text="", foreground="#555", wraplength=620, justify="left")
        self.info.pack(anchor="w", pady=(0, 10))

        self.b_run = self._big(right, "▶  Nhận diện", "Chạy nhận diện thời gian thực với hồ sơ này.", self.recognize)
        opts = ttk.Frame(right)
        opts.pack(anchor="w", padx=22, pady=(0, 8))
        self.gop = tk.BooleanVar(value=True)
        self.xn = tk.BooleanVar(value=True)
        self.gop_cb = ttk.Checkbutton(opts, text="Gộp bank của Đạt (khuyên dùng cho người lạ)", variable=self.gop)
        self.gop_cb.pack(anchor="w")
        ttk.Checkbutton(opts, text="Hỏi xác nhận khi không chắc", variable=self.xn).pack(anchor="w")

        self.b_day = self._big(right, "✎  Dạy từ mới", "Hiệu chỉnh nhanh: dạy vài mẫu (từ có sẵn hoặc từ mới) rồi dùng ngay, không train.", self.teach)
        self.b_thu = self._big(right, "●  Thu dữ liệu", "Ghi chú → thu → tự kiểm tra → hỏi thử nhận diện.", self.collect)
        self.b_kt = self._big(right, "✓  Kiểm tra dữ liệu", "Xem còn thiếu từ nào, kiểm tra chất lượng chi tiết.", self.check_detail)

        # ---- dưới: nhật ký
        bottom = ttk.Frame(main)
        bottom.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        main.rowconfigure(2, weight=2)
        bar = ttk.Frame(bottom)
        bar.pack(fill="x")
        self.status = ttk.Label(bar, text="Sẵn sàng.", font=("Segoe UI", 10, "bold"))
        self.status.pack(side="left")
        ttk.Button(bar, text="Xóa", command=lambda: self.log.delete("1.0", "end")).pack(side="right")
        self.stop_btn = ttk.Button(bar, text="■ Dừng", command=self.stop, state="disabled")
        self.stop_btn.pack(side="right", padx=6)
        self.log = scrolledtext.ScrolledText(bottom, height=12, font=("Consolas", 9), wrap="word")
        self.log.pack(fill="both", expand=True, pady=(4, 0))

    def _big(self, parent: ttk.Frame, text: str, hint: str, cmd: Callable[[], None]) -> ttk.Button:
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=(4, 2))
        b = ttk.Button(f, text=text, command=cmd, width=24)
        b.pack(side="left", ipady=8)
        ttk.Label(f, text=hint, foreground="#555", wraplength=380, justify="left").pack(side="left", padx=12)
        return b

    # ------------------------------------------------------------------ người ký
    def refresh_people(self, select: Optional[str] = None) -> None:
        """Vẽ danh sách NGAY bằng kết quả đã nhớ; người nào chưa có thì quét ở luồng nền (không làm đứng giao diện)."""
        if select:
            self.selected = select
        self._render_people()
        if any(dip.tinh_trang(p, chi_cache=True) is None for p in people.volunteer_ids()):
            self._scan_async()

    def _render_people(self) -> None:
        keep = self.selected
        for it in self.tree.get_children():
            self.tree.delete(it)
        n_dat, pending = 0, False
        for pid, v in people.load_people().items():
            if pid == people.OWNER_ID:
                label = f"{v.get('ten', pid)}  (chủ dự án)"
            else:
                t = dip.tinh_trang(pid, chi_cache=True)
                if t is None:
                    pending = True
                    label = f"{v.get('ten', pid)}  —  đang kiểm tra…"
                else:
                    n_dat += t["dat"]
                    label = f"{v.get('ten', pid)}  —  " + ("✓ đạt" if t["dat"] else f"thiếu ({t['tu_du']}/{t['tu_can']} từ)")
            self.tree.insert("", "end", iid=pid, text=label)
        self.progress.config(text=("đang kiểm tra dữ liệu…" if pending else
                                   f"{n_dat}/{dip.TARGET_NGUOI} tình nguyện viên đạt\n(5 = sơ bộ, {dip.TARGET_NGUOI} = kết luận đáng tin)"))
        if keep not in self.tree.get_children():
            keep = people.OWNER_ID
        self.tree.selection_set(keep)
        self.selected = keep
        self.update_info()

    def _scan_async(self) -> None:
        if self._scanning:
            self._rescan = True
            return
        self._scanning = True

        def work() -> None:
            try:
                for pid in people.volunteer_ids():
                    dip.tinh_trang(pid)
            finally:
                self.q.put(("SCANNED", 0))

        threading.Thread(target=work, daemon=True).start()

    def _on_select(self, _e: Any = None) -> None:
        sel = self.tree.selection()
        if sel:
            self.selected = sel[0]
            self.update_info()

    @property
    def pid(self) -> str:
        return self.selected

    def update_info(self) -> None:
        pid = self.pid
        st = people.profile_stats(pid)
        sk = people.skipped_words(pid)
        self.title_l.config(text=people.display_name(pid))
        if pid == people.OWNER_ID:
            self.info.config(text=f"Bank sản xuất của Đạt: {st['tu']} từ, {st['mau']} mẫu (đã đóng băng). Thu ở đây chỉ dành cho điều kiện môi trường.")
            self.gop_cb.state(["disabled"])
        else:
            t = dip.tinh_trang(pid, chi_cache=True)
            txt = f"{st['tu']} từ, {st['mau']} mẫu." + (f" Miễn: {', '.join(sk)}." if sk else "")
            if t is None:
                txt += "  Đang kiểm tra dữ liệu…"
            else:
                txt += "  ✓ Dữ liệu đạt." if t["dat"] else "  Còn thiếu: " + "; ".join(t["thieu"][:2])
            self.info.config(text=txt)
            self.gop_cb.state(["!disabled"])
        self.b_thu.config(text="●  Thu (điều kiện môi trường)" if pid == people.OWNER_ID else "●  Thu dữ liệu")

    def new_person(self) -> None:
        ten = simpledialog.askstring("Người mới", "Tên người ký (có dấu cũng được):", parent=self.root)
        if not ten or not ten.strip():
            return
        try:
            pid = dip.tao_ho_so(ten.strip())
        except ValueError as e:
            messagebox.showerror("Không tạo được", str(e), parent=self.root)
            return
        self.say(f"Đã tạo hồ sơ '{pid}'.")
        self.refresh_people(select=pid)

    # ------------------------------------------------------------------ chạy chương trình con
    def say(self, text: str) -> None:
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def run(self, args: List[str], title: str, on_done: Optional[Callable[[int], None]] = None) -> bool:
        if self.busy():
            messagebox.showwarning("Đang bận", "Đang chạy một việc khác. Bấm \"Dừng\" hoặc chờ nó xong.", parent=self.root)
            return False
        self.say(f"\n>>> py {' '.join(args)}")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen([sys.executable, "-u"] + args, cwd=dip.ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
        self.on_done = on_done
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.status.config(text=f"Đang chạy: {title}")
        self.stop_btn.config(state="normal")
        return True

    def _reader(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            self.q.put(("LINE", raw.decode("utf-8", errors="replace").rstrip("\r\n")))
        self.q.put(("DONE", proc.wait()))

    def _poll(self) -> None:
        n = 0
        try:
            while n < 200:                               # tối đa 200 dòng mỗi nhịp để giao diện luôn mượt
                kind, val = self.q.get_nowait()
                n += 1
                if kind == "LINE":
                    self.log.insert("end", val + "\n")
                elif kind == "DONE":
                    self.status.config(text=f"Xong (mã thoát {val}).")
                    self.stop_btn.config(state="disabled")
                    cb, self.on_done = self.on_done, None
                    self.refresh_people()
                    if cb:
                        self._after = (cb, val)              # chạy sau khi quét nền xong (nếu đang quét), không thì ngay
                        if not self._scanning:
                            self._run_after()
                elif kind == "SCANNED":
                    self._scanning = False
                    self._render_people()
                    self._run_after()
                    if self._rescan:
                        self._rescan = False
                        self.refresh_people()
        except queue.Empty:
            pass
        if n:
            self._trim_log()
            self.log.see("end")
        self.root.after(60 if n < 200 else 10, self._poll)

    def _run_after(self) -> None:
        if self._after is not None and not self._scanning:
            (cb, rc), self._after = self._after, None
            self.root.after(50, lambda: cb(rc))

    def _trim_log(self) -> None:
        if int(self.log.index("end-1c").split(".")[0]) > 3000:      # giữ nhật ký gọn: bỏ bớt dòng cũ
            self.log.delete("1.0", "1000.0")

    def stop(self) -> None:
        if self.busy():
            self.proc.terminate()
            self.say("(đã yêu cầu dừng)")

    def on_close(self) -> None:
        if self.busy() and not messagebox.askyesno("Thoát", "Đang chạy một việc. Dừng và thoát?", parent=self.root):
            return
        self.stop()
        self.root.destroy()

    # ------------------------------------------------------------------ các việc
    def recognize(self) -> None:
        pid = self.pid
        if pid != people.OWNER_ID and people.profile_stats(pid)["mau"] == 0:
            messagebox.showinfo("Chưa có dữ liệu", f"Hồ sơ '{pid}' chưa có mẫu. Bấm \"Thu dữ liệu\" trước.", parent=self.root)
            return
        templates = None
        if pid != people.OWNER_ID and self.gop.get():
            templates = dip.build_merged_bank(pid, True)
        self.run(dip.cmd_nhan_dien(pid, None, templates, self.xn.get()), f"Nhận diện ({people.display_name(pid)})")

    def teach(self) -> None:
        """Dạy từ: hỏi các từ + số mẫu, chạy hiệu chỉnh cho người đang chọn (Đạt thì dùng/tạo hồ sơ riêng, bank Đạt không bị sửa)."""
        pid = self.pid
        if pid == people.OWNER_ID:
            ten = simpledialog.askstring("Dạy từ", "Bank của Đạt được giữ nguyên. Từ dạy sẽ lưu vào hồ sơ riêng.\nTên hồ sơ (ví dụ Demo):",
                                         initialvalue="Demo", parent=self.root)
            if not ten or not ten.strip():
                return
            pid = people.resolve(ten.strip()) or ""
            if not pid:
                try:
                    pid = dip.tao_ho_so(ten.strip())
                except ValueError as e:
                    messagebox.showerror("Không tạo được", str(e), parent=self.root)
                    return
                self.refresh_people(select=pid)
        txt = simpledialog.askstring("Dạy từ", "Các từ cần dạy, cách nhau bằng dấu cách\n(từ có sẵn như HUNGRY, hoặc từ mới như XIN_CHAO):", parent=self.root)
        if not txt or not txt.split():
            return
        n = simpledialog.askinteger("Dạy từ", "Số mẫu mỗi từ:", initialvalue=3, minvalue=1, maxvalue=8, parent=self.root) or 3
        try:
            args = dip.lenh_hieu_chinh(pid, txt.split(), n)
        except ValueError as e:
            messagebox.showerror("Không dạy được", str(e), parent=self.root)
            return
        self.run(args, f"Dạy từ ({people.display_name(pid)})")

    def neg_record(self) -> None:
        n = simpledialog.askfloat("Quay mẫu âm", "Số phút (3–5). TUYỆT ĐỐI không ký trong lúc quay;\nlàm việc thường ngày theo gợi ý trên màn hình:",
                                  initialvalue=4.0, minvalue=1.0, maxvalue=15.0, parent=self.root)
        if n:
            self.run(dip.cmd_quay_mau_am(self.pid, n), "Quay mẫu âm")

    def neg_train(self) -> None:
        bo = simpledialog.askstring("Huấn luyện lại spotter",
                                    "Giữ lại phiên để kiểm tra độc lập: nhập một phần tên phiên (ví dụ tên phiên _neg mới nhất).\n"
                                    "Để trống = dùng hết dữ liệu. Model mới ghi ra spotter_model_new.pth (không đè model đang dùng):",
                                    parent=self.root)
        self.run(dip.cmd_huan_luyen_spotter([bo.strip()] if bo and bo.strip() else None), "Huấn luyện lại spotter")

    def neg_eval(self) -> None:
        models = ["spotter_model.pth"] + (["spotter_model_new.pth"] if os.path.exists(os.path.join(dip.ROOT, "spotter_model_new.pth")) else [])
        self.run(dip.cmd_cham_mau_am(None, models), "Chấm báo nhầm (mẫu âm)")

    def notes_dialog(self) -> Optional[Dict[str, str]]:
        d = tk.Toplevel(self.root)
        d.title("Ghi chú buổi thu")
        d.transient(self.root)
        d.grab_set()
        out: Dict[str, Optional[Dict[str, str]]] = {"v": None}
        frm = ttk.Frame(d, padding=14)
        frm.pack()
        vars_: Dict[str, Any] = {}
        spec = [("dominant_hand", "Tay thuận", ["Right", "Left"], "Right"), ("lighting", "Ánh sáng", ["sáng", "tối", "chói"], "sáng"),
                ("distance_cm_est", "Khoảng cách ngồi (cm)", None, ""), ("camera", "Camera (laptop, điện thoại...)", None, ""),
                ("notes", "Ghi chú khác", None, "")]
        for r, (key, label, opts, default) in enumerate(spec):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=3)
            v = tk.StringVar(value=default)
            vars_[key] = v
            if opts:
                ttk.Combobox(frm, textvariable=v, values=opts, state="readonly", width=24).grid(row=r, column=1, padx=8)
            else:
                ttk.Entry(frm, textvariable=v, width=27).grid(row=r, column=1, padx=8)

        def ok() -> None:
            out["v"] = {k: v.get().strip() for k, v in vars_.items()}
            d.destroy()

        btns = ttk.Frame(frm)
        btns.grid(row=len(spec), column=0, columnspan=2, pady=(10, 0))
        ttk.Button(btns, text="Bắt đầu thu", command=ok).pack(side="left", padx=4)
        ttk.Button(btns, text="Huỷ", command=d.destroy).pack(side="left", padx=4)
        self.root.wait_window(d)
        return out["v"]

    def collect(self) -> None:
        """Nút Thu: tình nguyện viên -> cả phiên (ghi chú, thu, kiểm tra, thử); Đạt -> chọn điều kiện môi trường."""
        if self.pid != people.OWNER_ID:
            self.volunteer_session()
            return
        menu = tk.Menu(self.root, tearoff=0)
        for c in dip.CONDITIONS:
            menu.add_command(label=f"Điều kiện: {c}", command=lambda cc=c: self.collect_condition(cc))
        b = self.b_thu
        menu.tk_popup(b.winfo_rootx(), b.winfo_rooty() + b.winfo_height())

    def volunteer_session(self, again: bool = False) -> None:
        pid = self.pid
        args = dip.cmd_thu(pid)
        session = args[args.index("--session") + 1]
        if not dip.notes_da_co(session):
            notes = self.notes_dialog()
            if notes is None:
                return
            dip.ghi_notes_row(pid, session, "normal", **notes)
        elif not again:
            self.say(f"(phiên {session} đã có ghi chú, thu tiếp vào cùng phiên)")
        self.run(args, f"Thu dữ liệu — {people.display_name(pid)}", on_done=lambda rc: self.after_collect(pid))

    def collect_condition(self, dieu_kien: str) -> None:
        args = dip.cmd_thu(people.OWNER_ID, dieu_kien)
        session = args[args.index("--session") + 1]
        if not dip.notes_da_co(session):
            notes = self.notes_dialog()
            if notes is None:
                return
            dip.ghi_notes_row(people.OWNER_ID, session, dieu_kien, **notes)
        self.run(args, f"Thu điều kiện {dieu_kien}")

    def after_collect(self, pid: str) -> None:
        t = dip.tinh_trang(pid, chi_cache=True) or dip.tinh_trang(pid)
        name = people.display_name(pid)
        if t["dat"]:
            self.say(f"✓ Dữ liệu của {name} ĐẠT ({t['tu_du']}/{t['tu_can']} từ, {t['mau']} mẫu).")
            if messagebox.askyesno("Đạt", f"Dữ liệu của {name} đạt.\nThử nhận diện ngay (bank Đạt + mẫu của {name}, có hỏi xác nhận)?", parent=self.root):
                self.run(dip.cmd_nhan_dien(pid, None, dip.build_merged_bank(pid, True), True), f"Nhận diện thử ({name})")
        else:
            self.say("Còn thiếu: " + "; ".join(t["thieu"]))
            if messagebox.askyesno("Còn thiếu", "Còn thiếu:\n- " + "\n- ".join(t["thieu"][:6]) + "\n\nThu bù ngay?", parent=self.root):
                self.volunteer_session(again=True)

    def check_detail(self) -> None:
        pid = self.pid
        if pid == people.OWNER_ID:
            messagebox.showinfo("Chọn tình nguyện viên", "Kiểm tra dữ liệu dành cho tình nguyện viên. Hãy chọn một người ở bên trái.", parent=self.root)
            return
        self.run(dip.cmd_kiem_tra(pid, people.skipped_words(pid)), f"Kiểm tra {people.display_name(pid)}")

    def skip_dialog(self) -> None:
        pid = self.pid
        if pid == people.OWNER_ID:
            return
        cur = " ".join(people.skipped_words(pid))
        s = simpledialog.askstring("Miễn từ", f"Các từ miễn cho {people.display_name(pid)} (cách nhau bằng dấu cách, để trống = không miễn):",
                                   initialvalue=cur, parent=self.root)
        if s is None:
            return
        people.set_skipped_words(pid, s.replace(",", " ").split())
        sk = people.skipped_words(pid)
        self.say(f"{people.display_name(pid)}: " + (f"miễn {', '.join(sk)}" if sk else "không miễn từ nào"))
        self.refresh_people()

    def evaluate(self, kind: str) -> None:
        signer = None
        if kind not in ("loso", "citizen"):
            if self.pid == people.OWNER_ID or people.profile_stats(self.pid)["mau"] == 0:
                messagebox.showinfo("Chọn người lạ", "Chọn một tình nguyện viên đã có dữ liệu ở bên trái.", parent=self.root)
                return
            signer = self.pid
        self.run(dip.cmd_danh_gia(kind, signer), f"Đánh giá: {kind}")


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
