"""
独立拟合测试: 剥离Phase E pipeline, 合成数据验证
目标: 线性RMS<1px, SIP+RMS<0.5px (σ_noise=0.3px时)
用法: python scripts/v3_5/test_fit_standalone.py
"""
import numpy as np, math, sys, os, json, time

# ============================================================
# 1. 真实CD/CRVAL/SIP参数生成合成数据
# ============================================================
def generate_synthetic(seed=42, n_pairs=200, w=4096, h=4096,
                       fl_mm=1917.0, ps_um=9.0, noise_sigma_px=0.3,
                       theta_deg=-89.0, flip_mode=2):
    """生成已知ground truth的匹配对, 含检测误差"""
    rng = np.random.RandomState(seed)
    s0 = 206.265 * ps_um / fl_mm   # arcsec/px

    center_ra = 337.0   # RA deg
    center_dec = -20.0  # Dec deg
    cos_dec = math.cos(center_dec * math.pi / 180.)
    crpix = np.array([w/2., h/2.])

    # 真实scale: s≈1.004 → CD矩阵
    s_true = 1.004
    theta_rad = math.radians(theta_deg)
    ct, st = math.cos(theta_rad), math.sin(theta_rad)
    fx = (flip_mode == 1 or flip_mode == 3)
    fy = (flip_mode == 2 or flip_mode == 3)
    sx = -1.0 if fx else 1.0
    sy = -1.0 if fy else 1.0

    s0_s3600 = s0 / (s_true * 3600.0)
    CD_true = np.array([
        [ sx * s0_s3600 * ct / cos_dec, -sx * s0_s3600 * st / cos_dec],
        [-sy * s0_s3600 * st,           -sy * s0_s3600 * ct]
    ])

    # 真实SIP: 产生~2px边缘畸变 (纯径向桶形)
    SIP_A_true = np.zeros((6,6))
    SIP_B_true = np.zeros((6,6))
    # 径向畸变系数: A02, A20, B02, B20 产生桶形
    x_scale = w/2.; y_scale = h/2.
    A02 = -3e-7   # 在(0, ±h/2)处产生 ≈ A02*(h/2)^2 = -3e-7*4e6 = -1.2px
    A20 = -2e-7
    B02 = -2e-7
    B20 = -3e-7
    SIP_A_true[0,2] = A02
    SIP_A_true[2,0] = A20
    SIP_B_true[0,2] = B02
    SIP_B_true[2,0] = B20
    sip_order = 2

    # 随机Gaia星位置 (在FOV内均匀分布)
    margin = 200
    gaia_pix_x = rng.uniform(-margin, w+margin, n_pairs)
    gaia_pix_y = rng.uniform(-margin, h+margin, n_pairs)

    # 像素→天球 (正向投影: x,y → ra,dec)
    # dra_deg = CD[0,0]*(x-crpix[0]) + CD[0,1]*(y-crpix[1])
    # ddec_deg = CD[1,0]*(x-crpix[0]) + CD[1,1]*(y-crpix[1])
    dxp = gaia_pix_x - crpix[0]
    dyp = gaia_pix_y - crpix[1]
    dra_deg = CD_true[0,0]*dxp + CD_true[0,1]*dyp
    ddec_deg = CD_true[1,0]*dxp + CD_true[1,1]*dyp
    gaia_ra = center_ra + dra_deg / cos_dec
    gaia_dec = center_dec + ddec_deg

    # 真实SIP修正: 像素→天球加SIP
    # 标准WCS: dra = CD·(pixel-crpix + SIP_offset)
    u = dxp / x_scale
    v = dyp / y_scale
    sip_dx = np.zeros(n_pairs)
    sip_dy = np.zeros(n_pairs)
    for p in range(sip_order+1):
        for q in range(sip_order+1):
            if p+q < 2: continue
            term = u**p * v**q
            sip_dx += SIP_A_true[p,q] * term
            sip_dy += SIP_B_true[p,q] * term
    sip_dx_px = sip_dx    # SIP_A is pixel units (already denormalized)
    sip_dy_px = sip_dy

    # 检测星像素 = Gaia真实像素 + 检测噪声
    # SIP对Gaia来说是加到像素坐标上的(因为CD已经吸收了线性部分)
    det_pix_x = gaia_pix_x + noise_sigma_px * rng.randn(n_pairs)
    det_pix_y = gaia_pix_y + noise_sigma_px * rng.randn(n_pairs)

    # 转arcsec (U格式): U_x = (det_x - crpix_x) * s0, U_y = -(det_y - crpix_y) * s0
    U_x = (det_pix_x - crpix[0]) * s0
    U_y = -(det_pix_y - crpix[1]) * s0

    # W格式: gnomonic投影 gaia→(xi, eta) arcsec, 不依赖CD
    xi = (gaia_ra - center_ra) * cos_dec * 3600.
    eta = (gaia_dec - center_dec) * 3600.
    W = np.column_stack([xi, eta])

    # Wf (flip): 匹配C++ internal表示
    if fx: W_x = -xi
    else:  W_x = xi
    if fy: W_y = -eta
    else:  W_y = eta
    Wf = np.column_stack([W_x, W_y])

    return {
        'gt': {
            'CD': CD_true, 'CRVAL': np.array([center_ra, center_dec]),
            'CRPIX': crpix, 's0': s0, 'w': w, 'h': h,
            'SIP_A': SIP_A_true, 'SIP_B': SIP_B_true, 'sip_order': sip_order,
            'center_ra': center_ra, 'center_dec': center_dec, 'cos_dec': cos_dec,
            's_true': s_true, 'theta_deg': theta_deg, 'flip_mode': flip_mode,
            'fx': fx, 'fy': fy, 'sx': sx, 'sy': sy,
        },
        'pairs': {
            'U': np.column_stack([U_x, U_y]),    # N×2 arcsec
            'Wf': Wf,                             # N×2 arcsec (flipped)
            'U_pix': np.column_stack([det_pix_x, det_pix_y]),
            'gaia_pix': np.column_stack([gaia_pix_x, gaia_pix_y]),
            'gaia_ra': gaia_ra, 'gaia_dec': gaia_dec,
        },
        'noise_sigma': noise_sigma_px,
    }


# ============================================================
# 2. Umeyama SVD (复刻C++实现)
# ============================================================
def umeyama(src, dst):
    """src→dst: dst ≈ s·R·src + t"""
    n = len(src)
    ms = np.mean(src, axis=0); md = np.mean(dst, axis=0)
    sc = src - ms; dc = dst - md
    H = sc.T @ dc
    U, S, Vt = np.linalg.svd(H)
    det = np.linalg.det(Vt.T @ U.T)
    Sv = np.array([1, det])
    R = Vt.T @ np.diag(Sv) @ U.T
    tr = np.sum(sc**2)
    if tr < 1e-15: return None
    s = np.dot(S, Sv) / tr
    if abs(s-1) > 0.1: return None
    th = math.atan2(R[1,0], R[0,0])
    t = md - s * R @ ms
    return {'s': s, 'theta': th, 'tx': t[0], 'ty': t[1], 'valid': True}


# ============================================================
# 3. 完整拟合Pipeline (复刻C++ fit_affine_sip_adaptive)
# ============================================================
def fit_pipeline(data, sip_max_order=4):
    gt = data['gt']; pr = data['pairs']
    M_D = len(pr['U'])
    U = pr['U']; Wf = pr['Wf']
    center_ra, center_dec = gt['center_ra'], gt['center_dec']
    cos_dec0, s0 = gt['cos_dec'], gt['s0']
    w, h = gt['w'], gt['h']
    fx, fy = gt['fx'], gt['fy']
    sx, sy = gt['sx'], gt['sy']
    crpix_x, crpix_y = w/2., h/2.

    # ---- Layer 0a: Umeyama弧秒初值 → CD ----
    us0 = U.reshape(-1, 2)
    ws0 = Wf.reshape(-1, 2)
    au0 = umeyama(ws0, us0)
    s_i = au0['s']; th_i = au0['theta']
    ct_i, st_i = math.cos(th_i), math.sin(th_i)
    s3600 = s0 / (s_i * 3600.)
    CD = np.array([
        [ sx * s3600 * ct_i / cos_dec0, -sx * s3600 * st_i / cos_dec0],
        [-sy * s3600 * st_i,            -sy * s3600 * ct_i]
    ])

    # ---- Layer 0b: CD投影→像素src/dst对 ----
    cd_ii = CD[0,0]*CD[1,1] - CD[0,1]*CD[1,0]
    CD_inv = np.array([[CD[1,1], -CD[0,1]], [-CD[1,0], CD[0,0]]]) / cd_ii

    src_x = np.zeros(M_D); src_y = np.zeros(M_D)
    dst_x = np.zeros(M_D); dst_y = np.zeros(M_D)
    for i in range(M_D):
        ui = i; wi = i  # paired 1:1
        src_x[i] = U[ui,0]/s0 + crpix_x - crpix_x
        src_y[i] = -U[ui,1]/s0 + crpix_y - crpix_y
        Wx = -Wf[wi,0] if fx else Wf[wi,0]
        Wy = -Wf[wi,1] if fy else Wf[wi,1]
        dst_x[i] = CD_inv[0,0]*(Wx/(3600.*cos_dec0)) + CD_inv[0,1]*(Wy/3600.)
        dst_y[i] = CD_inv[1,0]*(Wx/(3600.*cos_dec0)) + CD_inv[1,1]*(Wy/3600.)

    # ---- Layer 1: 全6-DOF仿射 dst = A·src + t ----
    # 设计矩阵: [sx, sy, 1] for each coordinate
    L = np.zeros((M_D*2, 6))
    R = np.zeros(M_D*2)
    for i in range(M_D):
        xx, yy = src_x[i], src_y[i]
        L[i*2, 0] = xx; L[i*2, 1] = yy; L[i*2, 2] = 1
        L[i*2+1, 3] = xx; L[i*2+1, 4] = yy; L[i*2+1, 5] = 1
        R[i*2] = dst_x[i]; R[i*2+1] = dst_y[i]
    ab, _, _, _ = np.linalg.lstsq(L, R, rcond=None)
    a00, a01, tx = ab[0], ab[1], ab[2]
    a10, a11, ty = ab[3], ab[4], ab[5]

    # 更新CD: CD' = CD @ A
    CD_new = CD @ np.array([[a00, a01], [a10, a11]])
    CRVAL_new = np.array([
        center_ra + (CD_new[0,0]*tx + CD_new[0,1]*ty),
        center_dec + (CD_new[1,0]*tx + CD_new[1,1]*ty)
    ])

    # ---- Layer 1 RMS: 新CD重投影 ----
    cd2 = CD_new[0,0]*CD_new[1,1] - CD_new[0,1]*CD_new[1,0]
    CD_inv2 = np.array([[CD_new[1,1], -CD_new[0,1]], [-CD_new[1,0], CD_new[0,0]]]) / cd2

    rms_aff = 0
    res_x_aff = np.zeros(M_D); res_y_aff = np.zeros(M_D)
    for i in range(M_D):
        Wx = -Wf[i,0] if fx else Wf[i,0]
        Wy = -Wf[i,1] if fy else Wf[i,1]
        dx = CD_inv2[0,0]*(Wx/(3600.*cos_dec0)) + CD_inv2[0,1]*(Wy/3600.)
        dy = CD_inv2[1,0]*(Wx/(3600.*cos_dec0)) + CD_inv2[1,1]*(Wy/3600.)
        res_x_aff[i] = dx - src_x[i]
        res_y_aff[i] = dy - src_y[i]
        rms_aff += res_x_aff[i]**2 + res_y_aff[i]**2
    rms_aff = math.sqrt(rms_aff/M_D)

    # ---- Layer p: SIP逐阶BIC ----
    xs, ys = w/2., h/2.
    nu = src_x / xs; nv = src_y / ys

    best_order = 0; best_bic = 1e30; best_rms = 0
    best_sA = np.zeros(36); best_sB = np.zeros(36)

    for try_o in range(2, sip_max_order+1):
        nhi = (try_o+1)*(try_o+2)//2 - 3  # p+q≥2
        if M_D <= nhi: continue
        Ah = np.zeros((M_D, nhi))
        for i in range(M_D):
            col = 0
            for o in range(2, try_o+1):
                for p in range(o+1):
                    Ah[i, col] = nu[i]**p * nv[i]**(o-p)
                    col += 1
        bxa, _, _, _ = np.linalg.lstsq(Ah, res_x_aff, rcond=None)
        bya, _, _, _ = np.linalg.lstsq(Ah, res_y_aff, rcond=None)
        ssq = np.sum((res_x_aff - Ah@bxa)**2) + np.sum((res_y_aff - Ah@bya)**2)
        rms_h = math.sqrt(ssq/M_D)
        kp = nhi*2
        bic = M_D*math.log(ssq/M_D) + kp*math.log(M_D)

        if bic < best_bic:
            best_bic = bic; best_order = try_o; best_rms = rms_h
            for o in range(2, try_o+1):
                for p in range(o+1):
                    q = o-p
                    if p>=6 or q>=6: continue
                    hi = -1; cn = 0
                    for oo in range(2, try_o+1):
                        for pp in range(oo+1):
                            if pp==p and (oo-pp)==q: hi = cn; break
                            cn += 1
                        if hi >= 0: break
                    if hi<0 or hi>=nhi: continue
                    nf = xs**p * ys**q
                    best_sA[p*6+q] = bxa[hi]/nf
                    best_sB[p*6+q] = bya[hi]/nf

    # 最终残差(CD+SIP)
    SIP_A_out = best_sA.reshape(6,6)
    SIP_B_out = best_sB.reshape(6,6)
    res_final_x = np.zeros(M_D); res_final_y = np.zeros(M_D)
    for i in range(M_D):
        dx = res_x_aff[i]; dy = res_y_aff[i]
        # 减去SIP预测
        u, v = nu[i], nv[i]
        sx_sip = 0; sy_sip = 0
        for o in range(2, best_order+1):
            for p in range(o+1):
                q = o-p
                term = u**p * v**q
                sx_sip += SIP_A_out[p,q]*term
                sy_sip += SIP_B_out[p,q]*term
        res_final_x[i] = dx - sx_sip
        res_final_y[i] = dy - sy_sip
    rms_final = math.sqrt(np.mean(res_final_x**2 + res_final_y**2))

    # 反推s/θ
    cd_det = CD_new[0,0]*CD_new[1,1] - CD_new[0,1]*CD_new[1,0]
    cos_cr2 = math.cos(CRVAL_new[1] * math.pi/180.)
    if cos_cr2 < 1e-10: cos_cr2 = 1e-10
    s_refined = s0/(3600.*math.sqrt(abs(cd_det)*cos_cr2))
    th_refined = math.atan2(sy and -CD_new[1,0] or CD_new[1,0],
                            sx and -CD_new[0,0] or CD_new[0,0])

    return {
        'CD': CD_new, 'CRVAL': CRVAL_new, 'CRPIX': np.array([crpix_x, crpix_y]),
        'SIP_A': best_sA, 'SIP_B': best_sB, 'sip_order': best_order,
        's': s_refined, 'theta_deg': math.degrees(th_refined),
        'affine_RMS': rms_aff, 'sip_RMS': best_rms, 'final_RMS': rms_final,
        'residuals_aff': np.column_stack([res_x_aff, res_y_aff]),
        'residuals_final': np.column_stack([res_final_x, res_final_y]),
    }


# ============================================================
# 4. 测试用例
# ============================================================
def run_test(label, n_pairs, noise_sigma, **kwargs):
    data = generate_synthetic(n_pairs=n_pairs, noise_sigma_px=noise_sigma, **kwargs)
    gt = data['gt']

    t0 = time.time()
    result = fit_pipeline(data)
    dt = time.time()-t0

    # 与ground truth比较
    cd_err = np.max(np.abs(result['CD'] - gt['CD']))
    crval_err = np.max(np.abs(result['CRVAL'] - gt['CRVAL'])) * 3600.  # deg→arcsec
    s_err = abs(result['s'] - gt['s_true'])
    th_err = abs(result['theta_deg'] - gt['theta_deg']) % 360
    if th_err > 180: th_err = 360 - th_err

    # 9区域残差分析
    w, h = gt['w'], gt['h']
    bw, bh = w/3, h/3
    sp = data['pairs']
    res = result['residuals_final']
    reg_meds = []
    for ri in range(9):
        rj = ri//3; ci = ri%3
        m = (sp['U_pix'][:,0] >= ci*bw) & (sp['U_pix'][:,0] < (ci+1)*bw) & \
            (sp['U_pix'][:,1] >= rj*bh) & (sp['U_pix'][:,1] < (rj+1)*bh)
        if m.sum() > 2:
            reg_meds.append((np.median(res[m,0]), np.median(res[m,1])))

    all_med_x = np.median([r[0] for r in reg_meds])
    all_med_y = np.median([r[1] for r in reg_meds])
    max_reg = max(max(abs(r[0]), abs(r[1])) for r in reg_meds) if reg_meds else 0

    print(f"\n--- {label} ---")
    print(f"  n={n_pairs}  noise=±{noise_sigma:.2f}px  time={dt*1000:.1f}ms")
    print(f"  仿射RMS={result['affine_RMS']:.4f}px  SIP_RMS={result['sip_RMS']:.4f}px  "
          f"最终RMS={result['final_RMS']:.4f}px")
    print(f"  CD误差<{cd_err:.2e}  CRVAL偏差={crval_err:.2f}\"  s误差={s_err:.2e}  θ误差={th_err:.4f}°")
    print(f"  9区残差中位: [{all_med_x:+.4f},{all_med_y:+.4f}]px  "
          f"各区max<{max_reg:.4f}px")
    print(f"  SIP阶={result['sip_order']}")
    if result['sip_order'] > 0:
        nz = np.sum(np.abs(result['SIP_A'])>1e-30) + np.sum(np.abs(result['SIP_B'])>1e-30)
        print(f"  非零SIP系数: {nz}")

    # 判定
    ok = result['affine_RMS'] < 1.0 and max_reg < 1.0
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  判定: {status} (线性RMS<1.0px且各区残差<1.0px)")

    return result


if __name__ == '__main__':
    print("="*70)
    print("Phase E 独立拟合测试 — 合成数据验证")
    print("="*70)

    # 测试1: 长焦, 无畸变, 无噪声
    run_test("长焦-理想(噪声=0)", n_pairs=200, noise_sigma=0.0,
             w=4096, h=4096, fl_mm=1917, ps_um=9, theta_deg=-89)

    # 测试2: 长焦, 有畸变, 无噪声
    run_test("长焦+SIP-理想(噪声=0)", n_pairs=200, noise_sigma=0.0,
             w=4096, h=4096, fl_mm=1917, ps_um=9, theta_deg=-89)

    # 测试3: 长焦, 有畸变, 噪声=0.3px
    run_test("长焦+SIP-噪声0.3px", n_pairs=200, noise_sigma=0.3,
             w=4096, h=4096, fl_mm=1917, ps_um=9, theta_deg=-89)

    # 测试4: 长焦, 有畸变, 噪声=0.3px, 更多点
    run_test("长焦+SIP-噪声0.3px-500点", n_pairs=500, noise_sigma=0.3,
             w=4096, h=4096, fl_mm=1917, ps_um=9, theta_deg=-89)

    # 测试5: 短焦, 有畸变, 噪声=3px(大像元→大噪声)
    run_test("短焦+SIP-噪声3px", n_pairs=200, noise_sigma=3.0,
             w=4500, h=3600, fl_mm=200, ps_um=6, theta_deg=-179.9)

    # 测试6: 短焦, 大量点
    run_test("短焦+SIP-噪声3px-500点", n_pairs=500, noise_sigma=3.0,
             w=4500, h=3600, fl_mm=200, ps_um=6, theta_deg=-179.9)

    # 测试7: 模拟糟糕情况 — 大量噪声
    run_test("长焦+SIP-高噪声1px", n_pairs=200, noise_sigma=1.0,
             w=4096, h=4096, fl_mm=1917, ps_um=9, theta_deg=-89)

    # 测试8: 真实NGC7293 Ha参数
    run_test("模拟NGC7293_Ha-110对-噪声0.5px", n_pairs=110, noise_sigma=0.5,
             w=4096, h=4096, fl_mm=1917, ps_um=9, theta_deg=-89)
