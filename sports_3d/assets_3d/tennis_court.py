"""
3D Tennis Court Visualization

Official ITF dimensions with origin at center of net.
Coordinate system:
  - X: across the court (sideline to sideline)
  - Y: up (height)
  - Z: along the court (positive = one baseline, negative = other baseline)

All measurements in meters.
"""

import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TennisCourtDimensions:
    """Official ITF tennis court dimensions in meters."""
    # Court length and width
    length: float = 23.77  # baseline to baseline
    singles_width: float = 8.23
    doubles_width: float = 10.97

    # Service box
    service_line_distance: float = 6.40  # from net

    # Net dimensions
    net_height_center: float = 0.914  # 3 feet
    net_height_posts: float = 1.07  # 3.5 feet

    # Derived measurements
    @property
    def half_length(self) -> float:
        return self.length / 2  # 11.885m from net to baseline

    @property
    def half_singles_width(self) -> float:
        return self.singles_width / 2  # 4.115m

    @property
    def half_doubles_width(self) -> float:
        return self.doubles_width / 2  # 5.485m

    @property
    def doubles_alley_width(self) -> float:
        return (self.doubles_width - self.singles_width) / 2  # 1.37m


class TennisCourt3D:
    """
    3D tennis court with origin at center of net.

    Stores court geometry as line segments for efficient rendering.
    """

    def __init__(self, include_doubles: bool = True):
        self.dims = TennisCourtDimensions()
        self.include_doubles = include_doubles
        self.lines: List[np.ndarray] = []
        self.line_colors: List[str] = []
        self._build_court()

    def _add_line(self, start: Tuple[float, float, float],
                  end: Tuple[float, float, float],
                  color: str = "white"):
        """Add a line segment to the court."""
        self.lines.append(np.array([start, end]))
        self.line_colors.append(color)

    def _add_rectangle(self, x_range: Tuple[float, float],
                       z_range: Tuple[float, float],
                       y: float = 0, color: str = "white"):
        """Add a rectangle (4 lines) to the court."""
        x1, x2 = x_range
        z1, z2 = z_range
        self._add_line((x1, y, z1), (x2, y, z1), color)
        self._add_line((x2, y, z1), (x2, y, z2), color)
        self._add_line((x2, y, z2), (x1, y, z2), color)
        self._add_line((x1, y, z2), (x1, y, z1), color)

    def _build_court(self):
        """Build all court lines."""
        d = self.dims

        # Baselines (at each end along Z axis)
        width = d.half_doubles_width if self.include_doubles else d.half_singles_width

        # Baseline 1 (positive Z side)
        self._add_line(
            (-width, 0, d.half_length),
            (width, 0, d.half_length)
        )

        # Baseline 2 (negative Z side)
        self._add_line(
            (-width, 0, -d.half_length),
            (width, 0, -d.half_length)
        )

        # Singles sidelines (along X axis)
        self._add_line(
            (-d.half_singles_width, 0, -d.half_length),
            (-d.half_singles_width, 0, d.half_length)
        )
        self._add_line(
            (d.half_singles_width, 0, -d.half_length),
            (d.half_singles_width, 0, d.half_length)
        )

        # Doubles sidelines (if included)
        if self.include_doubles:
            self._add_line(
                (-d.half_doubles_width, 0, -d.half_length),
                (-d.half_doubles_width, 0, d.half_length)
            )
            self._add_line(
                (d.half_doubles_width, 0, -d.half_length),
                (d.half_doubles_width, 0, d.half_length)
            )

        # Service lines (parallel to net, 6.4m from it on each side along Z)
        self._add_line(
            (-d.half_singles_width, 0, d.service_line_distance),
            (d.half_singles_width, 0, d.service_line_distance)
        )
        self._add_line(
            (-d.half_singles_width, 0, -d.service_line_distance),
            (d.half_singles_width, 0, -d.service_line_distance)
        )

        # Center service line (divides service boxes, along Z axis)
        self._add_line(
            (0, 0, d.service_line_distance),
            (0, 0, 0)  # to the net
        )
        self._add_line(
            (0, 0, -d.service_line_distance),
            (0, 0, 0)  # to the net
        )

        # Center marks on baselines (small perpendicular lines along X)
        center_mark_length = 0.1  # 10cm
        self._add_line(
            (-center_mark_length, 0, d.half_length),
            (center_mark_length, 0, d.half_length)
        )
        self._add_line(
            (-center_mark_length, 0, -d.half_length),
            (center_mark_length, 0, -d.half_length)
        )

        # Net line (at z=0, along X axis)
        self._add_line(
            (-d.half_doubles_width, 0, 0),
            (d.half_doubles_width, 0, 0),
            color="gray"
        )

    def get_all_lines_as_array(self) -> Tuple[np.ndarray, List[str]]:
        """
        Return all lines as a single array with shape (N, 2, 3)
        where N is number of line segments.
        """
        return np.array(self.lines), self.line_colors

    def get_court_surface_vertices(self) -> np.ndarray:
        """
        Return vertices for the court surface (for rendering as a filled quad).
        Returns shape (4, 3) for the corners.
        """
        d = self.dims
        width = d.half_doubles_width if self.include_doubles else d.half_singles_width
        return np.array([
            [-width, 0, -d.half_length],
            [width, 0, -d.half_length],
            [width, 0, d.half_length],
            [-width, 0, d.half_length],
        ])

    def get_surround_vertices(self, runoff_back: float = 6.4, runoff_side: float = 3.66):
        """
        Return vertices for the surround area (outside court lines).

        Args:
            runoff_back: Distance behind baseline (ITF minimum is 6.4m)
            runoff_side: Distance beside sidelines (ITF minimum is 3.66m)

        Returns:
            outer_corners: (4, 3) array for outer boundary
            inner_corners: (4, 3) array for inner boundary (court edge)
        """
        d = self.dims

        # Inner boundary (court lines)
        inner_half_x = d.half_doubles_width if self.include_doubles else d.half_singles_width
        inner_half_z = d.half_length

        # Outer boundary (with runoff)
        outer_half_x = inner_half_x + runoff_side
        outer_half_z = inner_half_z + runoff_back

        inner = np.array([
            [-inner_half_x, 0, -inner_half_z],
            [inner_half_x, 0, -inner_half_z],
            [inner_half_x, 0, inner_half_z],
            [-inner_half_x, 0, inner_half_z],
        ])

        outer = np.array([
            [-outer_half_x, 0, -outer_half_z],
            [outer_half_x, 0, -outer_half_z],
            [outer_half_x, 0, outer_half_z],
            [-outer_half_x, 0, outer_half_z],
        ])

        return outer, inner

    def to_plotly_figure(self,
                         inner_color: str = "#1A4B8C",  # Darker blue (inside lines)
                         outer_color: str = "#5BA3E0",  # Lighter blue (surround)
                         line_width: float = 4,
                         show_surface: bool = True,
                         runoff_back: float = 6.4,
                         runoff_side: float = 3.66) -> go.Figure:
        """
        Create a Plotly figure of the court.

        Args:
            inner_color: Color inside the court lines (darker blue)
            outer_color: Color of the surround/runoff area (lighter blue)
            line_width: Width of court lines
            show_surface: Whether to show the court surface
            runoff_back: Distance behind baseline for surround
            runoff_side: Distance beside sidelines for surround

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Add court surfaces (two-tone like Australian Open)
        if show_surface:
            outer, inner = self.get_surround_vertices(runoff_back, runoff_side)

            # Outer surround (lighter blue) - need to create a frame with hole
            # We'll do this with 4 rectangular strips around the inner court
            y_surf = -0.01  # Slightly below lines (Y is height)

            d = self.dims
            inner_hx = d.half_doubles_width if self.include_doubles else d.half_singles_width
            inner_hz = d.half_length
            outer_hx = inner_hx + runoff_side
            outer_hz = inner_hz + runoff_back

            # Back strip (behind positive Z baseline)
            back1 = np.array([
                [-outer_hx, y_surf, inner_hz],
                [outer_hx, y_surf, inner_hz],
                [outer_hx, y_surf, outer_hz],
                [-outer_hx, y_surf, outer_hz],
            ])
            fig.add_trace(go.Mesh3d(
                x=back1[:, 0], y=back1[:, 1], z=back1[:, 2],
                i=[0, 0], j=[1, 2], k=[2, 3],
                color=outer_color, opacity=1.0, showlegend=False, hoverinfo='skip'
            ))

            # Back strip (behind negative Z baseline)
            back2 = np.array([
                [-outer_hx, y_surf, -outer_hz],
                [outer_hx, y_surf, -outer_hz],
                [outer_hx, y_surf, -inner_hz],
                [-outer_hx, y_surf, -inner_hz],
            ])
            fig.add_trace(go.Mesh3d(
                x=back2[:, 0], y=back2[:, 1], z=back2[:, 2],
                i=[0, 0], j=[1, 2], k=[2, 3],
                color=outer_color, opacity=1.0, showlegend=False, hoverinfo='skip'
            ))

            # Side strip (positive X side)
            side1 = np.array([
                [inner_hx, y_surf, -inner_hz],
                [outer_hx, y_surf, -inner_hz],
                [outer_hx, y_surf, inner_hz],
                [inner_hx, y_surf, inner_hz],
            ])
            fig.add_trace(go.Mesh3d(
                x=side1[:, 0], y=side1[:, 1], z=side1[:, 2],
                i=[0, 0], j=[1, 2], k=[2, 3],
                color=outer_color, opacity=1.0, showlegend=False, hoverinfo='skip'
            ))

            # Side strip (negative X side)
            side2 = np.array([
                [-outer_hx, y_surf, -inner_hz],
                [-inner_hx, y_surf, -inner_hz],
                [-inner_hx, y_surf, inner_hz],
                [-outer_hx, y_surf, inner_hz],
            ])
            fig.add_trace(go.Mesh3d(
                x=side2[:, 0], y=side2[:, 1], z=side2[:, 2],
                i=[0, 0], j=[1, 2], k=[2, 3],
                color=outer_color, opacity=1.0, showlegend=False, hoverinfo='skip'
            ))

            # Inner court (darker blue)
            court_surface = self.get_court_surface_vertices()
            court_surface[:, 1] = y_surf  # Y is height
            fig.add_trace(go.Mesh3d(
                x=court_surface[:, 0],
                y=court_surface[:, 1],
                z=court_surface[:, 2],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color=inner_color,
                opacity=1.0,
                name="Court Surface",
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add all court lines
        for line, color in zip(self.lines, self.line_colors):
            fig.add_trace(go.Scatter3d(
                x=line[:, 0],
                y=line[:, 1],
                z=line[:, 2],
                mode='lines',
                line=dict(color=color, width=line_width),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Configure layout
        d = self.dims
        outer_hx = d.half_doubles_width + runoff_side
        outer_hz = d.half_length + runoff_back

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="X (across court) [m]",
                    range=[-outer_hx - 1, outer_hx + 1],
                    showgrid=True,
                    gridcolor='lightgray',
                ),
                yaxis=dict(
                    title="Y (height) [m]",
                    range=[-1, 10],
                    showgrid=True,
                    gridcolor='lightgray',
                ),
                zaxis=dict(
                    title="Z (along court) [m]",
                    range=[-outer_hz - 1, outer_hz + 1],
                    showgrid=True,
                    gridcolor='lightgray',
                ),
                aspectmode='data',
                bgcolor='#1a1a2e',
            ),
            title="Tennis Court - Origin at Net Center",
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='#1a1a2e',
        )

        return fig

    def add_ball_trajectory(self,
                           fig: go.Figure,
                           positions: np.ndarray,
                           color: str = "yellow",
                           size: float = 8,
                           name: str = "Ball") -> go.Figure:
        """
        Add a ball trajectory to an existing figure.

        Args:
            fig: Plotly figure (from to_plotly_figure)
            positions: Ball positions, shape (N, 3) for N time steps
            color: Ball color
            size: Ball marker size
            name: Legend name

        Returns:
            Updated figure
        """
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=size, color=color),
            name=name,
        ))

        return fig

    def add_ball_position(self,
                         fig: go.Figure,
                         position: np.ndarray,
                         color: str = "yellow",
                         size: float = 12,
                         name: str = "Ball") -> go.Figure:
        """Add a single ball position to the figure."""
        fig.add_trace(go.Scatter3d(
            x=[position[0]],
            y=[position[1]],
            z=[position[2]],
            mode='markers',
            marker=dict(size=size, color=color),
            name=name,
        ))
        return fig


def create_sample_ball_trajectory() -> np.ndarray:
    """
    Create a sample parabolic ball trajectory for testing.
    Simulates a ball served from one baseline to the other.
    """
    # Time steps
    t = np.linspace(0, 1.5, 50)

    # Starting position (near baseline, slightly to the side)
    # X: across court, Y: height, Z: along court
    x0, y0, z0 = 2.0, 2.5, 11.0  # Serve height ~2.5m

    # Initial velocity (toward other baseline along Z)
    vx0, vy0, vz0 = -3.0, 5.0, -25.0  # m/s

    # Gravity (acts on Y axis)
    g = 9.81

    # Compute trajectory (simple projectile motion)
    x = x0 + vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t**2  # Gravity on Y
    z = z0 + vz0 * t

    # Clip to positive y (ball doesn't go underground)
    y = np.maximum(y, 0)

    return np.column_stack([x, y, z])


# Quick dimension reference for your CV work
COURT_DIMS = {
    'length': 23.77,
    'singles_width': 8.23,
    'doubles_width': 10.97,
    'service_line_from_net': 6.40,
    'half_length': 11.885,
    'half_singles_width': 4.115,
    'half_doubles_width': 5.485,
    'net_height_center': 0.914,
    'net_height_posts': 1.07,
}


if __name__ == "__main__":
    # Create court
    court = TennisCourt3D(include_doubles=True)

    # Create figure
    fig = court.to_plotly_figure()

    # Add sample trajectory
    trajectory = create_sample_ball_trajectory()
    court.add_ball_trajectory(fig, trajectory)

    # Save to HTML (viewable in browser)
    # fig.write_html("/mnt/user-data/outputs/tennis_court_3d.html")
    # print("Saved to tennis_court_3d.html")

    # Also show if in interactive environment
    fig.show()
