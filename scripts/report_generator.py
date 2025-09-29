#!/usr/bin/env python3
"""
Report Generator for GT Baseball 6th Tool
Creates weekly/series summaries, CSV exports, and PDF reports.
"""

import pandas as pd
from fpdf import FPDF
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
from defensive_analytics import DefensiveAnalytics

class ReportGenerator:
    def __init__(self, df: pd.DataFrame, report_dir="reports"):
        self.df = df.copy()
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary(self):
        """Summarize key offensive, defensive, pitching and baserunning metrics."""
        summary = {}
        summary["Total Pitches"] = int(len(self.df))
        if "PitchVelo" in self.df:
            summary["Avg Pitch Velocity"] = round(self.df["PitchVelo"].dropna().mean(), 1)
            summary["Pitch Velo Std"] = round(self.df["PitchVelo"].dropna().std(), 2)
        else:
            summary["Avg Pitch Velocity"] = None
            summary["Pitch Velo Std"] = None

        if "ExitVelo" in self.df:
            summary["Avg Exit Velocity"] = round(self.df["ExitVelo"].dropna().mean(), 1)
        else:
            summary["Avg Exit Velocity"] = None

        if "LaunchAng" in self.df:
            summary["Avg Launch Angle"] = round(self.df["LaunchAng"].dropna().mean(), 1)
        else:
            summary["Avg Launch Angle"] = None

        # Strike rate (Strikes + Fouls) / total
        if "PitchOutcome" in self.df:
            strikes = self.df["PitchOutcome"].isin(["Strike", "Foul"]).sum()
            total = len(self.df)
            summary["Strike Rate %"] = round(strikes / total * 100, 1) if total > 0 else None
        else:
            summary["Strike Rate %"] = None

        # Hard hit rate among balls in play
        if all(col in self.df.columns for col in ["ExitVelo", "BallInPlay"]):
            bip = self.df[self.df["BallInPlay"] == True]
            if len(bip) > 0:
                hard_hits = (bip["ExitVelo"] > 95).sum()
                summary["Hard Hit Rate %"] = round(hard_hits / len(bip) * 100, 1)
            else:
                summary["Hard Hit Rate %"] = None
        else:
            summary["Hard Hit Rate %"] = None

        # Unique players
        if "PitcherName" in self.df:
            summary["Unique Pitchers"] = int(self.df["PitcherName"].nunique())
        else:
            summary["Unique Pitchers"] = None
        if "BatterName" in self.df:
            summary["Unique Batters"] = int(self.df["BatterName"].nunique())
        else:
            summary["Unique Batters"] = None

        # Defensive & baserunning metrics
        if "FielderReaction" in self.df:
            summary["Avg Fielder Reaction"] = round(self.df["FielderReaction"].dropna().mean(), 2)
        else:
            summary["Avg Fielder Reaction"] = None

        if "FielderRouteEfficiency" in self.df:
            summary["Avg Route Efficiency"] = round(self.df["FielderRouteEfficiency"].dropna().mean(), 2)
        else:
            summary["Avg Route Efficiency"] = None

        if "BatterTimeToFirst" in self.df:
            summary["Avg Time to First"] = round(self.df["BatterTimeToFirst"].dropna().mean(), 2)
        else:
            summary["Avg Time to First"] = None

        if "BaserunnerMaxSpeed" in self.df:
            summary["Avg Baserunner Max Speed"] = round(self.df["BaserunnerMaxSpeed"].dropna().mean(), 1)
        else:
            summary["Avg Baserunner Max Speed"] = None

        return pd.DataFrame([summary])

    def export_csv(self, filename="weekly_summary.csv"):
        """Export summary stats to CSV."""
        summary = self.generate_summary()
        csv_path = self.report_dir / filename
        summary.to_csv(csv_path, index=False)
        print(f"✅ CSV report saved: {csv_path}")

    def export_pdf(self, filename="weekly_report.pdf"):
        """Export a richer PDF report with summary stats and charts saved to disk."""
        summary = self.generate_summary().to_dict(orient="records")[0]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, "GT Baseball 6th Tool - Weekly Report", ln=True, align="C")
        pdf.ln(10)

        for key, value in summary.items():
            pdf.cell(200, 10, f"{key}: {value}", ln=True)

        pdf_path = self.report_dir / filename
        pdf.output(str(pdf_path))
        print(f"✅ PDF report saved: {pdf_path}")
        try:
            pdf_bytes = self.export_pdf_bytes()
            pdf_path = self.report_dir / filename
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            print(f"✅ PDF report saved: {pdf_path}")
        except Exception as e:
            print(f"❌ Failed to generate PDF: {e}")

    def export_pdf_bytes(self):
        """Return PDF bytes for the report (useful for in-memory downloads)."""
        summary = self.generate_summary().to_dict(orient="records")[0]

        # Create charts and embed them
        image_paths = self._create_charts()

        # Create defensive heatmap using DefensiveAnalytics if available
        try:
            da = DefensiveAnalytics(self.df)
            positioning = da.analyze_fielder_positioning()
            if 'error' not in positioning and len(positioning) > 0:
                # build dataframe: rows = metrics, cols = players
                metrics_df = pd.DataFrame(positioning).T
                # Only keep numeric columns
                numeric_df = metrics_df.select_dtypes(include=[float, int])
                if not numeric_df.empty:
                    plt.figure(figsize=(10, 4))
                    sns.heatmap(numeric_df.T, annot=True, fmt='.1f', cmap='RdYlGn')
                    plt.title('Fielding Performance Heatmap')
                    def_path = str(self.report_dir / 'grid_defense_heatmap.png')
                    plt.tight_layout()
                    plt.savefig(def_path)
                    plt.close()
                    image_paths.append(def_path)
        except Exception:
            pass

        # PDF subclass for consistent header/footer
        class ReportPDF(FPDF):
            def header(self_inner):
                self_inner.set_font('Arial', 'B', 14)
                self_inner.cell(0, 8, 'GT Baseball 6th Tool - Weekly Report', ln=True, align='C')
                self_inner.ln(2)

            def footer(self_inner):
                self_inner.set_y(-15)
                self_inner.set_font('Arial', 'I', 8)
                date_str = datetime.now().strftime('%Y-%m-%d')
                # left-aligned date
                self_inner.cell(0, 8, f'Report generated: {date_str}', align='L')
                # right-aligned page number
                self_inner.cell(0, 8, f'Page {self_inner.page_no()}', align='R')

        pdf = ReportPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Summary page
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Summary', ln=True)
        pdf.ln(2)

        pdf.set_font('Arial', '', 11)
        col1_w = 70
        for key, value in summary.items():
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(col1_w, 8, str(key), border=0)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 8, str(value))

        # Top pitchers table (if available)
        if 'PitcherName' in self.df.columns and 'PitchVelo' in self.df.columns:
            try:
                pstats = self.df.groupby('PitcherName').agg(
                    AvgVelo=('PitchVelo', 'mean'),
                    TotalPitches=('PitchVelo', 'count')
                ).round(2)
                if len(pstats) > 0:
                    pstats['AvgVelo'] = pstats['AvgVelo'].fillna(0)
                    top_p = pstats.sort_values('AvgVelo', ascending=False).head(10)
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 8, 'Top Pitchers', ln=True)
                    pdf.ln(2)

                    # Table header
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(60, 8, 'Pitcher', border=1)
                    pdf.cell(40, 8, 'Avg Velo', border=1)
                    pdf.cell(40, 8, 'Pitches', border=1, ln=1)

                    pdf.set_font('Arial', '', 10)
                    for idx, row in top_p.iterrows():
                        pdf.cell(60, 8, str(idx), border=1)
                        pdf.cell(40, 8, f"{row['AvgVelo']:.1f}", border=1)
                        pdf.cell(40, 8, str(int(row['TotalPitches'])), border=1, ln=1)
            except Exception:
                pass

        # Per-player sections: top pitchers and top batters
        per_player_images = []
        try:
            # Top pitchers by total pitches
            if 'PitcherName' in self.df.columns:
                pitch_counts = self.df['PitcherName'].value_counts().head(10).index.tolist()
                for p in pitch_counts:
                    imgs = self._create_player_charts(p, role='pitcher')
                    per_player_images.extend(imgs)

            # Top batters by balls in play or appearances
            if 'BatterName' in self.df.columns:
                if 'BallInPlay' in self.df.columns:
                    batter_counts = self.df[self.df['BallInPlay'] == True]['BatterName'].value_counts().head(10).index.tolist()
                else:
                    batter_counts = self.df['BatterName'].value_counts().head(10).index.tolist()
                for b in batter_counts:
                    imgs = self._create_player_charts(b, role='batter')
                    per_player_images.extend(imgs)
        except Exception:
            pass

        # Append per-player images to image_paths so they'll be embedded
        image_paths.extend(per_player_images)

        # Add images (one per page)
        for img in image_paths:
            try:
                if os.path.exists(img):
                    pdf.add_page()
                    pdf.image(img, x=10, y=20, w=190)
            except Exception:
                continue

        # Clean up generated images
        for img in image_paths:
            try:
                if os.path.exists(img):
                    os.remove(img)
            except Exception:
                pass

        pdf_str = pdf.output(dest='S')
        return pdf_str.encode('latin-1')
    
    def _create_charts(self):
        """Create PNG charts from the dataframe and return list of image file paths.

        Charts generated:
        - Pitch velocity histogram
        - Exit velocity histogram (balls in play)
        - Exit Velo vs Launch Angle scatter
        - Avg velocity by inning line
        - Pitch outcome pie chart (if available)
        """
        images = []
        try:
            # Create a grid for pitching/hitting overview: 2x2
            figs = []
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            # 1) Pitch velocity histogram
            ax = axes[0, 0]
            if 'PitchVelo' in self.df.columns:
                sns.histplot(self.df['PitchVelo'].dropna(), bins=20, kde=False, color='C0', ax=ax)
                ax.axvline(self.df['PitchVelo'].mean(), color='k', linestyle='--')
            ax.set_title('Pitch Velocity')

            # 2) Exit velocity (BIP)
            ax = axes[0, 1]
            if 'ExitVelo' in self.df.columns and 'BallInPlay' in self.df.columns:
                ev_df = self.df[self.df['BallInPlay'] == True]['ExitVelo'].dropna()
                if len(ev_df) > 0:
                    sns.histplot(ev_df, bins=15, color='C1', ax=ax)
            ax.set_title('Exit Velocity (BIP)')

            # 3) Exit Velo vs Launch Angle
            ax = axes[1, 0]
            if all(col in self.df.columns for col in ['ExitVelo', 'LaunchAng']):
                scatter_df = self.df.dropna(subset=['ExitVelo', 'LaunchAng'])
                if len(scatter_df) > 0:
                    ax.scatter(scatter_df['LaunchAng'], scatter_df['ExitVelo'], s=10)
            ax.set_title('Exit Velo vs Launch Angle')
            ax.set_xlabel('LaunchAng')
            ax.set_ylabel('ExitVelo')

            # 4) Avg velocity by inning
            ax = axes[1, 1]
            if 'Inning' in self.df.columns and 'PitchVelo' in self.df.columns:
                inning_summary = self.df.groupby('Inning')['PitchVelo'].mean().dropna()
                if len(inning_summary) > 0:
                    inning_summary.plot(marker='o', ax=ax)
            ax.set_title('Avg Pitch Velo by Inning')

            grid_path = str(self.report_dir / 'grid_overview.png')
            plt.tight_layout()
            plt.savefig(grid_path)
            plt.close()
            images.append(grid_path)

            # Secondary grid: outcomes, top pitchers, route/reaction
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            ax = axes[0, 0]
            if 'PitchOutcome' in self.df.columns:
                outcomes = self.df['PitchOutcome'].value_counts()
                outcomes.plot.pie(autopct='%1.1f%%', ax=ax)
                ax.set_ylabel('')
            ax.set_title('Pitch Outcomes')

            ax = axes[0, 1]
            if 'PitcherName' in self.df.columns and 'PitchVelo' in self.df.columns:
                pstats = self.df.groupby('PitcherName').agg(AvgVelo=('PitchVelo', 'mean')).dropna()
                if len(pstats) > 0:
                    top = pstats.sort_values('AvgVelo', ascending=False).head(8)
                    ax.bar(top.index, top['AvgVelo'])
                    ax.set_xticklabels(top.index, rotation=45, ha='right')
            ax.set_title('Top Pitchers Avg Velo')

            ax = axes[1, 0]
            if 'FielderRouteEfficiency' in self.df.columns:
                fre = self.df['FielderRouteEfficiency'].dropna()
                if len(fre) > 0:
                    sns.histplot(fre, bins=12, ax=ax, color='C3')
            ax.set_title('Route Efficiency')

            ax = axes[1, 1]
            if 'FielderReaction' in self.df.columns:
                fr = self.df['FielderReaction'].dropna()
                if len(fr) > 0:
                    sns.histplot(fr, bins=12, ax=ax, color='C4')
            ax.set_title('Reaction Time')

            grid2_path = str(self.report_dir / 'grid_secondary.png')
            plt.tight_layout()
            plt.savefig(grid2_path)
            plt.close()
            images.append(grid2_path)

            # Baserunning + pitches per inning compact
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ax = axes[0]
            if 'BaserunnerMaxSpeed' in self.df.columns:
                bs = self.df['BaserunnerMaxSpeed'].dropna()
                if len(bs) > 0:
                    sns.histplot(bs, bins=12, ax=ax, color='C6')
            ax.set_title('Baserunner Max Speed')

            ax = axes[1]
            if 'Inning' in self.df.columns:
                ipp = self.df.groupby('Inning').size()
                if len(ipp) > 0:
                    ax.bar(ipp.index.astype(str), ipp.values)
            ax.set_title('Pitches per Inning')

            grid3_path = str(self.report_dir / 'grid_baserunning_inning.png')
            plt.tight_layout()
            plt.savefig(grid3_path)
            plt.close()
            images.append(grid3_path)

        except Exception:
            pass

        return images

    def _create_player_charts(self, player_name: str, role: str = 'pitcher'):
        """Create charts for a specific player and return list of image paths.

        role: 'pitcher' or 'batter'
        Charts include velocity distribution, outcomes, and time-based trend if available.
        """
        imgs = []
        try:
            # Try to fetch defensive positioning data once to avoid repeated work
            try:
                da = DefensiveAnalytics(self.df)
                positioning = da.analyze_fielder_positioning()
            except Exception:
                positioning = {}

            has_defensive = isinstance(positioning, dict) and player_name in positioning

            if role == 'pitcher' and 'PitcherName' in self.df.columns:
                pdf = self.df[self.df['PitcherName'] == player_name]
                if len(pdf) == 0:
                    return imgs

                # If defensive data exists for this player, add a defensive subplot (4 panels); otherwise 3 panels
                n_panels = 4 if has_defensive else 3
                fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3))

                # Ensure axes is indexable
                if n_panels == 1:
                    axes = [axes]

                # 1) Velocity hist
                ax = axes[0]
                sns.histplot(pdf['PitchVelo'].dropna(), bins=15, color='C0', ax=ax)
                ax.set_title('Pitch Velo')

                # 2) Outcomes pie
                ax = axes[1]
                if 'PitchOutcome' in pdf.columns:
                    out = pdf['PitchOutcome'].value_counts()
                    if len(out) > 0:
                        ax.pie(out.values, labels=out.index, autopct='%1.1f%%')
                ax.set_title('Outcomes')

                # 3) Avg Velo by inning
                ax = axes[2]
                if 'Inning' in pdf.columns and 'PitchVelo' in pdf.columns:
                    try:
                        trend = pdf.groupby('Inning')['PitchVelo'].mean()
                        if len(trend) > 0:
                            ax.plot(trend.index, trend.values, marker='o')
                    except Exception:
                        pass
                ax.set_title('Avg Velo by Inning')

                # 4) Defensive metrics (optional)
                if has_defensive:
                    ax = axes[3]
                    try:
                        pdata = positioning.get(player_name, {})
                        # Select a few meaningful defensive metrics if present
                        metrics = {
                            'RouteEff': pdata.get('avg_route_efficiency'),
                            'Reaction': pdata.get('avg_reaction_time'),
                            'MaxSpeed': pdata.get('max_speed_achieved'),
                            'CatchProb': pdata.get('catch_probability_avg')
                        }
                        # Filter out None
                        metrics = {k: v for k, v in metrics.items() if v is not None}
                        if metrics:
                            ax.bar(metrics.keys(), [float(v) for v in metrics.values()], color='C5')
                    except Exception:
                        pass
                    ax.set_title('Defensive Metrics')

                # Add overall title (player name)
                fig.suptitle(str(player_name), fontsize=12)

                # Save combined image
                safe_name = str(player_name).replace(' ', '_')
                path_comb = str(self.report_dir / f'player_{safe_name}_combined.png')
                plt.tight_layout()
                # Move suptitle layout up so it doesn't overlap
                plt.subplots_adjust(top=0.85)
                plt.savefig(path_comb)
                plt.close()
                imgs.append(path_comb)

            if role == 'batter' and 'BatterName' in self.df.columns:
                bdf = self.df[self.df['BatterName'] == player_name]
                if len(bdf) == 0:
                    return imgs

                # If defensive data exists for this player, add a defensive subplot (4 panels); otherwise 3 panels
                n_panels = 4 if has_defensive else 3
                fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3))
                if n_panels == 1:
                    axes = [axes]

                # 1) Exit Velo
                ax = axes[0]
                if 'ExitVelo' in bdf.columns:
                    ev = bdf[bdf['BallInPlay'] == True]['ExitVelo'].dropna() if 'BallInPlay' in bdf.columns else bdf['ExitVelo'].dropna()
                    if len(ev) > 0:
                        sns.histplot(ev, bins=12, color='C1', ax=ax)
                ax.set_title('Exit Velo')

                # 2) Launch Angle
                ax = axes[1]
                if 'LaunchAng' in bdf.columns:
                    la = bdf[bdf['BallInPlay'] == True]['LaunchAng'].dropna() if 'BallInPlay' in bdf.columns else bdf['LaunchAng'].dropna()
                    if len(la) > 0:
                        sns.histplot(la, bins=12, color='C2', ax=ax)
                ax.set_title('Launch Angle')

                # 3) Baserunner speed
                ax = axes[2]
                if 'BaserunnerMaxSpeed' in bdf.columns:
                    bs = bdf['BaserunnerMaxSpeed'].dropna()
                    if len(bs) > 0:
                        sns.histplot(bs, bins=12, color='C3', ax=ax)
                ax.set_title('Max Speed')

                # 4) Defensive metrics (optional)
                if has_defensive:
                    ax = axes[3]
                    try:
                        pdata = positioning.get(player_name, {})
                        metrics = {
                            'RouteEff': pdata.get('avg_route_efficiency'),
                            'Reaction': pdata.get('avg_reaction_time'),
                            'MaxSpeed': pdata.get('max_speed_achieved'),
                            'CatchProb': pdata.get('catch_probability_avg')
                        }
                        metrics = {k: v for k, v in metrics.items() if v is not None}
                        if metrics:
                            ax.bar(metrics.keys(), [float(v) for v in metrics.values()], color='C5')
                    except Exception:
                        pass
                    ax.set_title('Defensive Metrics')

                # Add overall title (player name)
                fig.suptitle(str(player_name), fontsize=12)

                # Save combined image
                safe_name = str(player_name).replace(' ', '_')
                path_comb = str(self.report_dir / f'player_{safe_name}_combined.png')
                plt.tight_layout()
                plt.subplots_adjust(top=0.85)
                plt.savefig(path_comb)
                plt.close()
                imgs.append(path_comb)

        except Exception:
            pass

        return imgs
