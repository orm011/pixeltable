from typing import Any, Dict
import pytest
import numpy as np

import pixeltable as pt
from pixeltable.type_system import IntType
from pixeltable.tests.utils import create_test_tbl, make_tbl, create_table_data, assert_resultset_eq
import pixeltable.exceptions as exc


class TestSnapshot:
    def run_basic_test(
            self, cl: pt.Client, tbl: pt.Table, snap: pt.Table, extra_items: Dict[str, Any], filter: Any,
            reload_md: bool
    ) -> None:
        tbl_path, snap_path = cl.get_path(tbl), cl.get_path(snap)
        # run the initial query against the base table here, before reloading, otherwise the filter breaks
        tbl_select_list = [tbl[col_name] for col_name in tbl.column_names()]
        tbl_select_list.extend([value_expr for _, value_expr in extra_items.items()])
        orig_resultset = tbl.select(*tbl_select_list).where(filter).order_by(tbl.c2).collect()

        if reload_md:
            # reload md
            cl = pt.Client(reload=True)
            tbl = cl.get_table(tbl_path)
            snap = cl.get_table(snap_path)

        # view select list: base cols followed by view cols
        snap_select_list = [snap[col_name] for col_name in snap.column_names()[len(extra_items):]]
        snap_select_list.extend([snap[col_name] for col_name in extra_items.keys()])
        snap_query = snap.select(*snap_select_list).order_by(snap.c2)
        r1 = list(orig_resultset)
        r2 = list(snap_query.collect())
        assert_resultset_eq(snap_query.collect(), orig_resultset)

        # adding data to a base table doesn't change the snapshot
        rows = list(tbl.select(tbl.c1, tbl.c1n, tbl.c2, tbl.c3, tbl.c4, tbl.c5, tbl.c6, tbl.c7).collect())
        status = tbl.insert(rows)
        assert status.num_rows == len(rows)
        assert_resultset_eq(snap_query.collect(), orig_resultset)

        # update() doesn't affect the view
        status = tbl.update({'c3': tbl.c3 + 1.0})
        assert status.num_rows == tbl.count()
        assert_resultset_eq(snap_query.collect(), orig_resultset)

        # delete() doesn't affect the view
        num_tbl_rows = tbl.count()
        status = tbl.delete()
        assert status.num_rows == num_tbl_rows
        assert_resultset_eq(snap_query.collect(), orig_resultset)

        tbl.revert()  # undo delete()
        tbl.revert()  # undo update()
        tbl.revert()  # undo insert()
        # can't revert a version referenced by a snapshot
        with pytest.raises(exc.Error) as excinfo:
            tbl.revert()
        assert 'version is needed' in str(excinfo.value)

        # can't drop a table with snapshots
        with pytest.raises(exc.Error) as excinfo:
            cl.drop_table(tbl_path)
        assert snap_path in str(excinfo.value)

        cl.drop_table(snap_path)
        cl.drop_table(tbl_path)

    def test_basic(self, test_client: pt.Client) -> None:
        cl = test_client
        cl.create_dir('main')
        cl.create_dir('snap')
        tbl_path = 'main.tbl1'
        snap_path = 'snap.snap1'

        for reload_md in [False, True]:
            for has_filter in [False, True]:
                for has_cols in [False, True]:
                    cl = pt.Client(reload=True)
                    tbl = create_test_tbl(name=tbl_path, client=cl)
                    schema = {
                        'v1': tbl.c3 * 2.0,
                        # include a lambda to make sure that is handled correctly
                        'v2': {'value': lambda c3: c3 * 2.0, 'type': pt.FloatType()}
                    } if has_cols else {}
                    extra_items = {'v1': tbl.c3 * 2.0, 'v2': tbl.c3 * 2.0} if has_cols else {}
                    filter = tbl.c2 < 10 if has_filter else None
                    snap = cl.create_view(snap_path, tbl, schema=schema, filter=filter, is_snapshot=True)
                    self.run_basic_test(cl, tbl, snap, extra_items=extra_items, filter=filter, reload_md=reload_md)

    def test_views_of_snapshots(self, test_client: pt.Client) -> None:
        cl = test_client
        t = cl.create_table('tbl', {'a': IntType()})
        rows = [{'a': 1}, {'a': 2}, {'a': 3}]
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0
        s1 = cl.create_view('s1', t, is_snapshot=True)
        v1 = cl.create_view('v1', s1, is_snapshot=False)
        s2 = cl.create_view('s2', v1, is_snapshot=True)
        v2 = cl.create_view('v2', s2, is_snapshot=False)

        def verify(s1: pt.Table, s2: pt.Table, v1: pt.Table, v2: pt.Table) -> None:
            assert s1.count() == len(rows)
            assert v1.count() == len(rows)
            assert s2.count() == len(rows)
            assert v2.count() == len(rows)

        verify(s1, s2, v1, v2)

        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0
        verify(s1, s2, v1, v2)

        cl = pt.Client(reload=True)
        s1 = cl.get_table('s1')
        s2 = cl.get_table('s2')
        v1 = cl.get_table('v1')
        v2 = cl.get_table('v2')
        verify(s1, s2, v1, v2)

    def test_snapshot_of_view_chain(self, test_client: pt.Client) -> None:
        cl = test_client
        t = cl.create_table('tbl', {'a': IntType()})
        rows = [{'a': 1}, {'a': 2}, {'a': 3}]
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0
        v1 = cl.create_view('v1', t, is_snapshot=False)
        v2 = cl.create_view('v2', v1, is_snapshot=False)
        s = cl.create_view('s', v2, is_snapshot=True)

        def verify(v1: pt.Table, v2: pt.Table, s: pt.Table) -> None:
            assert v1.count() == t.count()
            assert v2.count() == t.count()
            assert s.count() == len(rows)

        verify(v1, v2, s)

        status = t.insert(rows)
        assert status.num_rows == len(rows) * 3  # we also updated 2 views
        assert status.num_excs == 0
        verify(v1, v2, s)

        cl = pt.Client(reload=True)
        v1 = cl.get_table('v1')
        v2 = cl.get_table('v2')
        s = cl.get_table('s')
        verify(v1, v2, s)

    def test_multiple_snapshot_paths(self, test_client: pt.Client) -> None:
        cl = test_client
        t = create_test_tbl(cl)
        c4 = t.select(t.c4).order_by(t.c2).collect().to_pandas()['c4']
        orig_c3 = t.select(t.c3).collect().to_pandas()['c3']
        v = cl.create_view('v', base=t, schema={'v1': t.c3 + 1})
        s1 = cl.create_view('s1', v, is_snapshot=True)
        t.drop_column('c4')
        # s2 references the same view version as s1, but a different version of t (due to a schema change)
        s2 = cl.create_view('s2', v, is_snapshot=True)
        t.update({'c6': {'a': 17}})
        # s3 references the same view version as s2, but a different version of t (due to a data change)
        s3 = cl.create_view('s3', v, is_snapshot=True)
        t.update({'c3': t.c3 + 1})
        # s4 references different versions of t and v
        s4 = cl.create_view('s4', v, is_snapshot=True)

        def validate(t: pt.Table, v: pt.Table, s1: pt.Table, s2: pt.Table, s3: pt.Table, s4: pt.Table) -> None:
            # c4 is only visible in s1
            assert np.all(s1.select(s1.c4).collect().to_pandas()['c4'] == c4)
            with pytest.raises(AttributeError):
                _ = t.select(t.c4).collect()
            with pytest.raises(AttributeError):
                _ = v.select(v.c4).collect()
            with pytest.raises(AttributeError):
                _ = s2.select(s2.c4).collect()
            with pytest.raises(AttributeError):
                _ = s3.select(s3.c4).collect()
            with pytest.raises(AttributeError):
                _ = s4.select(s4.c4).collect()

            # c3
            assert np.all(t.select(t.c3).order_by(t.c2).collect().to_pandas()['c3'] == orig_c3 + 1)
            assert np.all(s1.select(s1.c3).order_by(s1.c2).collect().to_pandas()['c3'] == orig_c3)
            assert np.all(s2.select(s2.c3).order_by(s2.c2).collect().to_pandas()['c3'] == orig_c3)
            assert np.all(s3.select(s3.c3).order_by(s3.c2).collect().to_pandas()['c3'] == orig_c3)
            assert np.all(s4.select(s4.c3).order_by(s4.c2).collect().to_pandas()['c3'] == orig_c3 + 1)

            # v1
            assert np.all(
                v.select(v.v1).order_by(v.c2).collect().to_pandas()['v1'] == \
                t.select(t.c3).order_by(t.c2).collect().to_pandas()['c3'] + 1)
            assert np.all(s1.select(s1.v1).order_by(s1.c2).collect().to_pandas()['v1'] == orig_c3 + 1)
            assert np.all(s2.select(s2.v1).order_by(s2.c2).collect().to_pandas()['v1'] == orig_c3 + 1)
            assert np.all(s3.select(s3.v1).order_by(s3.c2).collect().to_pandas()['v1'] == orig_c3 + 1)
            assert np.all(
                s4.select(s4.v1).order_by(s4.c2).collect().to_pandas()['v1'] == \
                t.select(t.c3).order_by(t.c2).collect().to_pandas()['c3'] + 1)

        validate(t, v, s1, s2, s3, s4)

        # make sure it works after metadata reload
        cl = pt.Client(reload=True)
        t, v = cl.get_table('test_tbl'), cl.get_table('v')
        s1, s2, s3, s4 = cl.get_table('s1'), cl.get_table('s2'), cl.get_table('s3'), cl.get_table('s4')
        validate(t, v, s1, s2, s3, s4)
