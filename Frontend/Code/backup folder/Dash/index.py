import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
import flask
import glob
import os


care_logo = "Logo.png"


#image_directory = 'c:/wamp64/www/fyp'
#logo_images = [os.path.basename(x) for x in glob.glob('{}Logo.png'.format(image_directory))]
#static_image_route = '/static/'

nav_items = dbc.Row(
    dbc.Nav(
    [
        dbc.Col(dbc.NavItem(dbc.NavLink("Calculator", href="#", active=True))),
        dbc.Col(dbc.NavItem(dbc.NavLink("Dashboard", href="#"))),
    ]),
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

age_input = dbc.FormGroup(
    [
        dbc.Label("Age:", html_for="Age"),
        dbc.Input(type="text", id="age", placeholder="Enter age"),
        #dbc.FormText("e.g. 50", color="secondary",),
    ]
)

gender_input = dbc.FormGroup(
    [
        dbc.Label("Gender:"),
        dbc.RadioItems(
            options=[
                {"label": "Male", "value": "male"},
                {"label": "Female", "value": "female"},
            ],
            value=1,
            id="gender",
            inline=True,
        ),
    ]
)

her2_input = dbc.FormGroup(
    [
        dbc.Label("HER2:"),
        dbc.RadioItems(
            options=[
                {"label": "Positive", "value": "positive"},
                {"label": "Negative", "value": "negative"},
            ],
            value=1,
            id="her2",
            inline=True,
        ),
    ]
)

er_input = dbc.FormGroup(
    [
        dbc.Label("ER:"),
        dbc.RadioItems(
            options=[
                {"label": "Positive", "value": "positive"},
                {"label": "Negative", "value": "negative"},
            ],
            value=1,
            id="er",
            inline=True,
        ),
    ]
)

pr_input = dbc.FormGroup(
    [
        dbc.Label("PR:"),
        dbc.RadioItems(
            options=[
                {"label": "Positive", "value": "positive"},
                {"label": "Negative", "value": "negative"},
            ],
            value=1,
            id="pr",
            inline=True,
        ),
    ]
)

cerb_input = dbc.FormGroup(
    [
        dbc.Label("CERB-B2:"),
        dbc.RadioItems(
            options=[
                {"label": "Positive", "value": "positive"},
                {"label": "Negative", "value": "negative"},
            ],
            value=1,
            id="cerb",
            inline=True,
        ),
    ]
)

tumour_input = dbc.FormGroup(
    [
        dbc.Label("Tumour Size:"),
        dbc.Select(
            id="tumour_size",
            options=[
                {"label": "TX", "value": 'TX'},
                {"label": "T0", "value": 'T0'},
                {"label": "T1", "value": 'T1'},
                {"label": "T2", "value": 'T2'},
                {"label": "T3", "value": 'T3'},
                {"label": "T4", "value": 'T4'},            
                ],
        )
    ]
)

lymph_input = dbc.FormGroup(
    [
        dbc.Label("Lymph Nodes:"),
        dbc.Select(
            id="lymph_nodes",
            options=[
                {"label": "N0", "value": 'N0'},
                {"label": "N1", "value": 'N1'},
                {"label": "N2", "value": 'N2'},
                {"label": "N3", "value": 'N3'},
                ],
        )
    ]
)

metastasis_input = dbc.FormGroup(
    [
        dbc.Label("Metastasis:"),
        dbc.Select(
            id="metastasis",
            options=[
                {"label": "MX", "value": 'MX'},
                {"label": "M1", "value": 'M0'},
                {"label": "M1", "value": 'M1'},
                {"label": "M2", "value": 'M2'},
                {"label": "M3", "value": 'M3'},
                ],
        )
    ]
)


navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img('Logo.png', height="30px")),
                    dbc.Col(dbc.NavbarBrand("  C.A.R.E", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://plot.ly",
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(nav_items, id="navbar-collapse", navbar=True),

    ],
    color="dark",
    dark=True,
)

body = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Calculator"),
                    ],
                    md=4, 
                )
            ]
        ),

        html.Br(),
        html.Br(),

        dbc.Form([

            dbc.Row(
                [
                    dbc.Col(
                        [
                            age_input
                        ], md=4,
                    ),

                    dbc.Col(md=4),

                    dbc.Col(
                        [
                            gender_input
                        ]
                    )
                ]
            ),

            dbc.Row(
                [
                    dbc.Col(
                        [
                            tumour_input
                        ], md=4,
                    ),

                    dbc.Col(md=4),

                    dbc.Col(
                        [
                            her2_input
                        ]
                    )
                ]
            ),

            dbc.Row(
                [
                    dbc.Col(
                        [
                            lymph_input
                        ], md=4,
                    ),

                    dbc.Col(md=4),

                    dbc.Col(
                        [
                            er_input
                        ]
                    )
                ]
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        [
                            metastasis_input
                        ], md=4,
                    ),

                    dbc.Col(md=4),

                    dbc.Col(
                        [
                            pr_input
                        ]
                    )
                ]
            ),

            dbc.Row(
                [
                    dbc.Col(
                        [
                            
                        ], md=4,
                    ),

                    dbc.Col(md=4),

                    dbc.Col(
                        [
                            cerb_input
                        ]
                    )
                ]
            ),

            html.Br(),

            dbc.Row(
                [   
                    dbc.Col(
                        [
                        ], md=5
                    ),

                    dbc.Col(
                        [
                            dbc.Button("Submit", color="primary"),
                        ], md=5
                    )
                ]
            ),

        ])

    ],
    className="mt-4",
)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([navbar, body])


@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)

if __name__ == "__main__":
    app.run_server()

# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


